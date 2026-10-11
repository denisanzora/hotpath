"""Packaging metadata. A mis-nested table in pyproject.toml makes every `pip install` fail while all source-tree tests
still pass (the 0.2 candidate shipped exactly that), so the metadata is checked structurally and, where available,
parsed with a real TOML parser."""
import os
import re
import unittest

import hotpath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        raw = f.read()
    try:
        import tomllib                      # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib         # optional on 3.10
        except ImportError:
            return None, raw.decode("utf-8")
    return tomllib.loads(raw.decode("utf-8")), raw.decode("utf-8")


class PyprojectTests(unittest.TestCase):
    def test_parsed_structure(self):
        data, _text = _pyproject()
        if data is None:
            self.skipTest("no TOML parser on this Python; structural test below applies")
        proj = data["project"]
        self.assertEqual(proj["version"], hotpath.__version__)
        self.assertIsInstance(proj["classifiers"], list)                          # not swallowed by another table
        self.assertEqual(set(proj["optional-dependencies"]), {"schema"})          # only extras, each a list of PEP 508 strings
        for reqs in proj["optional-dependencies"].values():
            self.assertTrue(all(re.match(r"^[A-Za-z0-9_.\-]+\s*(\[.*\])?\s*([<>=!~]=?.*)?$", r) for r in reqs), reqs)
        self.assertEqual(proj["dependencies"], [])                                # the core stays dependency-free
        self.assertEqual(proj["scripts"], {"hotpath": "hotpath.cli:main", "hotpath-mcp": "hotpath.mcp_server:main"})

    def test_keys_stay_inside_their_tables(self):
        # works without a parser: every key of [project] must appear before the next [table] header
        _data, text = _pyproject()
        sections, current = {}, None
        for line in text.splitlines():
            m = re.match(r"^\[([^\]]+)\]\s*$", line)
            if m:
                current = m.group(1)
                sections[current] = []
            elif current and re.match(r"^[A-Za-z][A-Za-z0-9_\-]*\s*=", line):
                sections[current].append(line.split("=")[0].strip())
        for key in ("name", "version", "description", "readme", "license", "requires-python", "dependencies", "keywords", "classifiers"):
            self.assertIn(key, sections["project"], "%s must be a key of [project]" % key)
        self.assertEqual(sections["project.optional-dependencies"], ["schema"])

    def test_console_entry_points_import(self):
        from hotpath import cli, mcp_server
        self.assertTrue(callable(cli.main) and callable(mcp_server.main))


if __name__ == "__main__":
    unittest.main()
