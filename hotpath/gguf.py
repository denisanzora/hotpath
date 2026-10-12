"""GGUF header reader: metadata, per-tensor shapes and types, and bytes-per-token estimates.

Reads only the header (a few MB), never the weights. Tensor types without a known name are
reported as typeN; their byte size is taken from the gap to the next tensor's offset.
"""
import struct
from collections import Counter

GG_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
            12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
            20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
            29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}

# bytes per block and block size for types whose layout we know; used for exact byte counts
BLOCK_BYTES = {"F32": (4, 1), "F16": (2, 1), "BF16": (2, 1), "Q8_0": (34, 32), "Q4_0": (18, 32), "Q4_K": (144, 256),
               "Q6_K": (210, 256), "TQ1_0": (54, 256)}

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class _Reader:
    def __init__(self, f):
        self.f = f

    def rd(self, fmt):
        return struct.unpack("<" + fmt, self.f.read(struct.calcsize(fmt)))

    def s(self):
        (n,) = self.rd("Q")
        return self.f.read(n).decode("utf-8", "replace")

    def skip(self, t):
        if t == 8:
            (n,) = self.rd("Q")
            self.f.seek(n, 1)
        elif t == 9:
            (et,) = self.rd("I")
            (n,) = self.rd("Q")
            if et in _SCALAR:
                self.f.seek(n * struct.calcsize(_SCALAR[et]), 1)
            else:
                for _ in range(n):
                    self.skip(et)
        else:
            self.f.seek(struct.calcsize(_SCALAR[t]), 1)

    def val(self, t, keep_arrays=8):
        if t == 8:
            return self.s()
        if t == 9:
            (et,) = self.rd("I")
            (n,) = self.rd("Q")
            vals = [self.val(et, keep_arrays) for _ in range(min(n, keep_arrays))]
            # elements past the kept head (tokenizer vocabularies: 100k+ strings) are skipped, not decoded
            if n > keep_arrays:
                if et in _SCALAR:
                    self.f.seek((n - keep_arrays) * struct.calcsize(_SCALAR[et]), 1)
                else:
                    for _ in range(n - keep_arrays):
                        self.skip(et)
            return {"array_len": n, "head": vals}
        return self.rd(_SCALAR[t])[0]


def read_header(path):
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        (ver,) = r.rd("I")
        (nt,) = r.rd("Q")
        (nkv,) = r.rd("Q")
        meta = {}
        for _ in range(nkv):
            k = r.s()
            (t,) = r.rd("I")
            meta[k] = r.val(t)
        tensors = []
        for _ in range(nt):
            name = r.s()
            (nd,) = r.rd("I")
            dims = list(r.rd("Q" * nd))
            (ty,) = r.rd("I")
            (off,) = r.rd("Q")
            tensors.append({"name": name, "dims": dims, "type_id": ty, "type": GG_TYPES.get(ty, "type%d" % ty), "offset": off})
    tensors.sort(key=lambda t: t["offset"])
    for i, t in enumerate(tensors):
        n = 1
        for d in t["dims"]:
            n *= d
        t["n_elements"] = n
        bb = BLOCK_BYTES.get(t["type"])
        if bb:
            t["bytes"] = n // bb[1] * bb[0]
        elif i + 1 < len(tensors):
            t["bytes"] = tensors[i + 1]["offset"] - t["offset"]
        else:
            t["bytes"] = None
    return {"version": ver, "n_tensors": nt, "n_kv": nkv, "meta": meta, "tensors": tensors}


def summarize(hdr):
    meta = hdr["meta"]
    arch = meta.get("general.architecture")

    def mk(key, default=None):
        return meta.get("%s.%s" % (arch, key), default)

    by_type = Counter()
    bytes_by_type = Counter()
    for t in hdr["tensors"]:
        by_type[t["type"]] += 1
        bytes_by_type[t["type"]] += t["bytes"] or 0
    total = sum(bytes_by_type.values())
    embd = sum(t["bytes"] or 0 for t in hdr["tensors"] if t["name"] == "token_embd.weight")
    output = sum(t["bytes"] or 0 for t in hdr["tensors"] if t["name"] == "output.weight")
    n_layer = mk("block_count")
    n_head_kv = mk("attention.head_count_kv")
    head_k = mk("attention.key_length") or (mk("embedding_length") // mk("attention.head_count") if mk("attention.head_count") else None)
    head_v = mk("attention.value_length") or head_k
    full_interval = mk("full_attention_interval")
    n_attn_layers = n_layer // full_interval if (n_layer and full_interval) else n_layer
    kv_bytes_per_token_f16 = None
    if n_attn_layers and n_head_kv and head_k and head_v:
        kv_bytes_per_token_f16 = n_attn_layers * n_head_kv * (head_k + head_v) * 2
    streamed = total - embd  # embeddings are gathered per token, not streamed
    return {
        "architecture": arch,
        "n_layer": n_layer,
        "n_embd": mk("embedding_length"),
        "n_ff": mk("feed_forward_length"),
        "n_head": mk("attention.head_count"),
        "n_head_kv": n_head_kv,
        "head_dim_k": head_k,
        "n_vocab": (meta.get("tokenizer.ggml.tokens") or {}).get("array_len"),
        "n_ctx_train": mk("context_length"),
        "full_attention_interval": full_interval,
        "n_full_attention_layers": n_attn_layers,
        "tensor_types": {k: {"count": by_type[k], "bytes": bytes_by_type[k]} for k in by_type},
        "total_bytes": total,
        "token_embd_bytes": embd,
        "output_weight_bytes": output,
        "weight_bytes_streamed_per_decode_token": streamed,
        "kv_bytes_per_token_f16": kv_bytes_per_token_f16,
    }


def decode_ceiling_tps(summary, bandwidth_gbs, n_past=0):
    """Bandwidth-bound ceiling for batch-1 decode: bytes read per token / bandwidth."""
    if not bandwidth_gbs:
        return None
    per_token = summary["weight_bytes_streamed_per_decode_token"]
    if summary.get("kv_bytes_per_token_f16") and n_past:
        per_token += summary["kv_bytes_per_token_f16"] * n_past
    return round(bandwidth_gbs * 1e9 / per_token, 1)
