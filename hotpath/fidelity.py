"""Output fidelity between experiment arms, from untimed probe requests that return top-k log-probabilities.

Timed requests never ask for probabilities (it slows generation measurably), so after an arm's timed requests the runner
sends every prompt once more with `n_probs` and the same greedy settings. This module compares those probes.

What it can and cannot say:
- `logprob-identical`: same tokens and the same top-k log-probabilities at every position (within float noise).
- `token-identical`: same tokens, but the log-probabilities moved; how far is reported.
- `diverged-near-tie`: the outputs differ from one token on, and at that token the baseline's choice and the candidate's
  were within NEAR_TIE_NATS of each other. This is evidence that the change shifted a numerically close decision; it is
  NOT evidence that the different continuation is acceptable.
- `diverged`: the outputs differ and the decision was not close (or its margin is unknown because the candidate's token was
  outside the baseline's top-k).
The KL figure covers only the tokens both arms reported in their top-k, renormalized; it is not a full-distribution KL.
"""
import math

NEAR_TIE_NATS = 0.1          # log-probability gap at the first divergent token below which the choice counts as a near-tie
LOGPROB_EQUAL_NATS = 1e-6    # differences below this are float noise in the server's JSON
IDENTICAL = ("logprob-identical", "token-identical")


def probe_from_response(resp):
    """{tokens, top} from a llama-server /completion response that was asked for n_probs."""
    cp = resp.get("completion_probabilities") or []
    top = []
    for pos in cp:
        cands = pos.get("top_logprobs")
        if cands is None:                      # older servers: probs with prob values
            cands = [{"id": x.get("id"), "logprob": math.log(x["prob"]) if x.get("prob") else None} for x in pos.get("probs") or []]
        top.append([[c.get("id"), c.get("logprob")] for c in cands if c.get("id") is not None and c.get("logprob") is not None])
    tokens = resp.get("tokens") or [pos.get("id") for pos in cp]
    return {"tokens": list(tokens), "top": top, "k": max((len(t) for t in top), default=0)}


def probe_from_chat_response(resp):
    """{tokens, top} from an OpenAI-compatible /v1/chat/completions response asked for `logprobs` and `top_logprobs`
    (choices[0].logprobs.content). Token ids where the server returns them (llama-server does), else token strings."""
    ch = (resp.get("choices") or [{}])[0]
    content = ((ch.get("logprobs") or {}).get("content")) or []
    tokens, top = [], []
    for pos in content:
        tokens.append(pos.get("id", pos.get("token")))
        top.append([[c.get("id", c.get("token")), c.get("logprob")] for c in pos.get("top_logprobs") or []
                    if c.get("id", c.get("token")) is not None and c.get("logprob") is not None])
    return {"tokens": tokens, "top": top, "k": max((len(t) for t in top), default=0)}


def _lp(pos):
    return {(int(t) if isinstance(t, (int, float)) else t): float(lp) for t, lp in pos}


def _partial_kl(pb, pc):
    shared = set(pb) & set(pc)
    if not shared:
        return None
    zb = sum(math.exp(pb[t]) for t in shared)
    zc = sum(math.exp(pc[t]) for t in shared)
    if zb <= 0 or zc <= 0:
        return None
    kl = 0.0
    for t in shared:
        b = math.exp(pb[t]) / zb
        c = math.exp(pc[t]) / zc
        if b > 0 and c > 0:
            kl += b * math.log(b / c)
    return max(0.0, kl)


def _margin(pos_lp, chosen, other):
    """(margin, at_least): how much more likely `chosen` was than `other` at this position; when `other` is not in the
    reported top-k the exact margin is unknown and only a lower bound is given."""
    if chosen not in pos_lp:
        return None, None
    if other in pos_lp:
        return pos_lp[chosen] - pos_lp[other], None
    return None, pos_lp[chosen] - min(pos_lp.values())


def compare_probes(base, cand):
    """One prompt: how the candidate's probe output relates to the baseline's."""
    bt, ct = base.get("tokens") or [], cand.get("tokens") or []
    n = min(len(bt), len(ct))
    d = next((i for i in range(n) if bt[i] != ct[i]), None)
    if d is None and len(bt) != len(ct):
        d = n
    upto = n if d is None else d
    shift, kls = 0.0, []
    for i in range(min(upto, len(base.get("top") or []), len(cand.get("top") or []))):
        pb, pc = _lp(base["top"][i]), _lp(cand["top"][i])
        for t in set(pb) & set(pc):
            shift = max(shift, abs(pb[t] - pc[t]))
        kl = _partial_kl(pb, pc)
        if kl is not None:
            kls.append(kl)
    k = min(base.get("k") or 0, cand.get("k") or 0) or None
    out = {"tokens_compared": n, "first_divergence_index": d, "max_logprob_shift": round(shift, 6) if shift > LOGPROB_EQUAL_NATS else 0.0,
           "max_partial_kl": round(max(kls), 6) if kls else None,
           "kl_basis": "KL over the top-%s tokens both arms reported at each position before any divergence, renormalized; not the full distribution" % (k or "k"),
           "baseline_margin_nats": None, "baseline_margin_at_least_nats": None, "candidate_margin_nats": None}
    if d is None:
        out["class"] = "logprob-identical" if out["max_logprob_shift"] == 0.0 else "token-identical"
        return out
    if d >= n or d >= len(base.get("top") or []) or d >= len(cand.get("top") or []):
        out.update(cascade(bt, ct, d))
        out["class"] = "diverged"
        out["note"] = "the outputs differ in length; no probabilities at the divergence"
        return out
    out.update(cascade(bt, ct, d))
    pb, pc = _lp(base["top"][d]), _lp(cand["top"][d])
    mb, mb_least = _margin(pb, bt[d], ct[d])
    mc, _ = _margin(pc, ct[d], bt[d])
    out.update({"baseline_margin_nats": round(mb, 4) if mb is not None else None,
                "baseline_margin_at_least_nats": round(mb_least, 4) if mb_least is not None else None,
                "candidate_margin_nats": round(mc, 4) if mc is not None else None})
    near = mb is not None and mb <= NEAR_TIE_NATS and (mc is None or mc <= NEAR_TIE_NATS)
    out["class"] = "diverged-near-tie" if near else "diverged"
    return out


def cascade(bt, ct, d):
    """How far a divergence carried: tokens generated after it by each arm, and how much of the rest still lines up
    (difflib ratio of the two remaining token sequences: 1.0 the same tokens re-aligned, near 0 a different continuation).
    A sequence measure, not a semantic one: similar tokens can still mean a different answer."""
    import difflib
    rb, rc = list(bt[d:]), list(ct[d:])
    sim = difflib.SequenceMatcher(None, rb, rc, autojunk=False).ratio() if (rb or rc) else 1.0
    return {"tokens_after_divergence": [len(rb), len(rc)], "rest_similarity": round(sim, 3),
            "divergence_at_fraction": round(d / max(1, len(bt)), 3)}


def is_equivalent(prompt_result):
    return prompt_result.get("class") in IDENTICAL


def classify_arm(prompts, validity_dropped=False, baseline_reproducible=True, policy=None, policy_check=None):
    """One candidate arm against the baseline, over every compared prompt. `equivalent` is the only thing that may permit
    automatic adoption: identical tokens on every prompt, or differences that an operator-configured policy accepted after
    it was actually checked (policy_check: {"passed": bool, "basis": str})."""
    sampled = sum(1 for p in prompts.values() if p.get("class") == "sampled")
    classes = [p.get("class") for p in prompts.values() if p.get("class") != "sampled"]
    counts = {c: classes.count(c) for c in sorted(set(classes))}
    base = {"prompts": len(classes), "counts": counts, "policy": policy, "policy_check": policy_check, "sampled_not_judged": sampled}
    if not classes and sampled and not validity_dropped:
        return dict(base, **{"class": "not judged (sampled requests)", "equivalent": False,
                             "why": "all %d compared requests sample (temperature above 0): their outputs are not expected to repeat, so identity "
                                    "says nothing; validity and decision fields are the evidence" % sampled})
    if validity_dropped:
        return dict(base, **{"class": "invalid or regressed", "equivalent": False,
                             "why": "output validity dropped; a decision-level regression whatever the tokens"})
    if not baseline_reproducible:
        return dict(base, **{"class": "baseline not reproducible", "equivalent": False,
                             "why": "the baseline's own repeated outputs differ, so identity between arms says nothing"})
    if classes and all(c in IDENTICAL for c in classes):
        cls = "logprob-identical" if all(c == "logprob-identical" for c in classes) else "token-identical"
        return dict(base, **{"class": cls, "equivalent": True, "why": "identical tokens on all %d prompts" % len(classes)})
    if policy and policy_check and policy_check.get("passed"):
        return dict(base, **{"class": "different, verified by policy", "equivalent": True,
                             "why": "outputs differ on %d of %d prompts; the configured policy %r passed: %s" % (
                                 sum(1 for c in classes if c not in IDENTICAL), len(classes), policy, policy_check.get("basis"))})
    diverged = [c for c in classes if c not in IDENTICAL]
    if not classes:
        return dict(base, **{"class": "not compared", "equivalent": False, "why": "no probe outputs to compare"})
    if all(c == "diverged-near-tie" for c in diverged):
        return dict(base, **{"class": "numerically close, behavior differs", "equivalent": False,
                             "why": "outputs differ on %d of %d prompts, each from a near-tie token on; the continuations differ and were not verified" % (
                                 len(diverged), len(classes))})
    return dict(base, **{"class": "unverified differences", "equivalent": False,
                         "why": "outputs differ on %d of %d prompts, at least one at a clear (or unknown) margin" % (len(diverged), len(classes))})
