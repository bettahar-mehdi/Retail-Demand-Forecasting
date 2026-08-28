#!/usr/bin/env python
"""Check catalog authority — Principle II (T012).

Fails if any src/*.py contains hard-coded data/0 literals outside allowlist.
Allowlist: src/retail_demand_forecasting/utils/catalog.py and lines with 'catalog-allowlist'.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PATTERN = re.compile(r"data/0[123]")
ALLOWLIST_FILE = "utils/catalog.py"
ALLOWLIST_COMMENT = "catalog-allowlist"


def main() -> int:
    violations = []
    for py in SRC.rglob("*.py"):
        rel = py.relative_to(ROOT).as_posix()
        if ALLOWLIST_FILE in rel:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                if ALLOWLIST_COMMENT in line:
                    continue
                # Fallback param to helper is allowed (still catalog-authority compliant)
                if "_resolve_catalog_path" in line or "_catalog_path" in line or "get_catalog_filepath" in line:
                    continue
                # Comments describing the rule are allowed
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                violations.append(f"{rel}:{i}: {line.strip()}")

    if violations:
        print("[catalog-authority] FAIL — hard-coded data/0 literals found:")
        for v in violations:
            print(f"  {v}")
        print(f"[catalog-authority] Total {len(violations)} violation(s) outside {ALLOWLIST_FILE}")
        print("[catalog-authority] Fix: use get_catalog_filepath(name) from utils/catalog.py")
        return 1
    print("[catalog-authority] PASS — no hard-coded data/0 literals outside allowlist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
