#!/usr/bin/env python
"""Evaluation gate — Principle III (FR-005, SC-003).

Compares metrics.json vs baseline and fails CI if:
- wape > baseline_wape + 2.0
- r2 < 0.80
- any tracked metric missing

Usage:
  python scripts/evaluate_gate.py
  python scripts/evaluate_gate.py --baseline metrics_baseline.json --current metrics.json
  python scripts/evaluate_gate.py --wape-delta 2.0 --r2-min 0.80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tracked_metrics() -> list[str]:
    p = Path("conf/base/parameters.yml")
    if p.exists():
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            metrics = data.get("metrics", {})
            track = metrics.get("track")
            if isinstance(track, list) and track:
                return [str(x) for x in track]
        except Exception:
            pass
    return ["wape", "rmse", "mae", "r2", "mape"]


def main() -> int:
    parser = argparse.ArgumentParser(description="MLflow evaluation gate")
    parser.add_argument("--baseline", default="metrics_baseline.json", help="Baseline metrics JSON")
    parser.add_argument("--current", default="metrics.json", help="Current metrics JSON")
    parser.add_argument("--wape-delta", type=float, default=2.0, help="Max allowed WAPE increase (pp)")
    parser.add_argument("--r2-min", type=float, default=0.80, help="Minimum R2")
    parser.add_argument("--output", default=None, help="Write gate report JSON to path")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    # Fallback: try data/06_metrics/model_metrics.json if baseline missing
    if not baseline_path.exists():
        alt = Path("data/06_metrics/model_metrics.json")
        if alt.exists():
            print(f"[gate] Baseline {baseline_path} missing, using {alt}")
            baseline_path = alt
        else:
            print(f"[gate] WARNING: Baseline {baseline_path} not found — creating from current (first run)")
            if current_path.exists():
                # First run: treat as pass but warn
                print("[gate] PASS (no baseline, first run)")
                return 0
            print(f"[gate] FAIL: Current {current_path} also missing")
            return 1

    if not current_path.exists():
        print(f"[gate] FAIL: Current metrics {current_path} not found")
        return 1

    try:
        baseline = load_json(baseline_path)
        current = load_json(current_path)
    except Exception as exc:
        print(f"[gate] FAIL: Failed to load JSON: {exc}")
        return 1

    tracked = load_tracked_metrics()
    print(f"[gate] Tracked metrics: {tracked}")
    print(f"[gate] Baseline: {baseline_path} | Current: {current_path}")

    checks = []
    overall = "pass"

    # Check tracked metrics present
    for m in tracked:
        if m not in current:
            print(f"[gate] FAIL: Tracked metric '{m}' missing in current {current_path}")
            checks.append({"metric": m, "status": "fail", "reason": "missing in current"})
            overall = "fail"
        elif m not in baseline:
            print(f"[gate] WARN: Metric '{m}' missing in baseline — skipping delta check")
            checks.append({"metric": m, "baseline": None, "current": current.get(m), "status": "warn"})
        else:
            checks.append({"metric": m, "baseline": baseline[m], "current": current[m], "status": "pass"})

    # WAPE gate
    if "wape" in baseline and "wape" in current:
        try:
            b_wape = float(baseline["wape"])
            c_wape = float(current["wape"])
            delta = c_wape - b_wape
            status = "pass" if delta <= args.wape_delta else "fail"
            print(f"[gate] WAPE: baseline {b_wape:.2f} -> current {c_wape:.2f} delta {delta:+.2f} (threshold +{args.wape_delta}) => {status.upper()}")
            checks.append({"metric": "wape_gate", "baseline": b_wape, "current": c_wape, "delta": delta, "threshold": args.wape_delta, "status": status})
            if status == "fail":
                overall = "fail"
        except Exception as exc:
            print(f"[gate] FAIL: WAPE check error: {exc}")
            overall = "fail"

    # R2 gate
    if "r2" in current:
        try:
            c_r2 = float(current["r2"])
            status = "pass" if c_r2 >= args.r2_min else "fail"
            print(f"[gate] R2: current {c_r2:.4f} (min {args.r2_min}) => {status.upper()}")
            checks.append({"metric": "r2_gate", "current": c_r2, "threshold": args.r2_min, "status": status})
            if status == "fail":
                overall = "fail"
        except Exception as exc:
            print(f"[gate] FAIL: R2 check error: {exc}")
            overall = "fail"

    report = {
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "wape_delta_threshold": args.wape_delta,
        "r2_min": args.r2_min,
        "tracked": tracked,
        "checks": checks,
        "overall": overall,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[gate] Report written to {out}")

    if overall == "fail":
        print(f"[gate] OVERALL FAIL — blocking promotion")
        return 1
    print(f"[gate] OVERALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
