"""Catalog authority helper — single source for dataset filepaths (Principle II).

All dataset paths MUST be declared in ``conf/base/catalog.yml`` and resolved
via this helper, never hard-coded as ``data/01_raw/...`` literals elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    # src/retail_demand_forecasting/utils/catalog.py -> repo root (3 levels up)
    return Path(__file__).resolve().parents[3]


def _load_catalog() -> dict[str, Any]:
    """Load and merge catalog.yml (base + local overlay if present)."""
    base = _repo_root() / "conf" / "base" / "catalog.yml"
    local = _repo_root() / "conf" / "local" / "catalog.yml"
    data: dict[str, Any] = {}
    if base.exists():
        with open(base, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                data.update(loaded)
    if local.exists():
        try:
            with open(local, encoding="utf-8") as f:
                overlay = yaml.safe_load(f) or {}
                if isinstance(overlay, dict):
                    data.update(overlay)
        except Exception:
            pass
    return data


def get_catalog_filepath(name: str) -> Path:
    """Return absolute Path for dataset ``name`` per catalog.yml.

    Raises KeyError if name not in catalog, ValueError if entry has no filepath.
    """
    catalog = _load_catalog()
    if name not in catalog:
        raise KeyError(f"Dataset '{name}' not found in conf/base/catalog.yml")
    entry = catalog[name]
    if not isinstance(entry, dict) or "filepath" not in entry:
        raise ValueError(f"Catalog entry '{name}' has no 'filepath' key: {entry}")
    fp = str(entry["filepath"])
    # Resolve relative to repo root; if already absolute, keep as is
    p = Path(fp)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve() if p.exists() else _repo_root() / Path(fp)


def list_catalog_names() -> list[str]:
    """List all dataset names in the catalog."""
    return list(_load_catalog().keys())
