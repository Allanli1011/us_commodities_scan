"""Diagnose why MACD divergence is so rare — break down rule-pass counts and
print every contract's best (loose/strict) attempt, regardless of score.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_fetcher import load_prices
from src.indicators.macd import detect_triple_divergence
from src.universe import load_universe


def main() -> None:
    universe = load_universe()
    tickers = universe["ticker"].tolist()
    names = dict(zip(universe["ticker"], universe["name"]))

    by_passed = Counter()
    failed_rule_counter = Counter()
    best_per_ticker: dict[str, tuple[str, int, str, list[str]]] = {}

    for t in tickers:
        df = load_prices(t)
        if df is None or df.empty:
            continue
        for direction in ("top", "bottom"):
            res = detect_triple_divergence(df, direction=direction)
            if not res.rule_checks:
                continue
            key = res.n_passed
            by_passed[key] += 1
            for rc in res.failed_rules:
                failed_rule_counter[rc.code] += 1
            cur_best = best_per_ticker.get(t)
            if cur_best is None or res.n_passed > cur_best[1]:
                failed_codes = [rc.code for rc in res.failed_rules]
                best_per_ticker[t] = (direction, res.n_passed, res.hit_kind, failed_codes)

    print("\n── Rule pass distribution (all ticker × direction combos) ──")
    for k in sorted(by_passed, reverse=True):
        print(f"  {k}/5 passed: {by_passed[k]}")

    print("\n── Failed-rule frequency ──")
    for rule, n in failed_rule_counter.most_common():
        print(f"  {rule}: {n}")

    print("\n── Best attempt per contract (sorted by passed count) ──")
    rows = sorted(best_per_ticker.items(), key=lambda kv: -kv[1][1])
    for t, (d, n_pass, kind, failed) in rows:
        print(f"  {t:6s} {names.get(t, ''):28s} {d:6s}  {n_pass}/5  {kind:5s}  miss={','.join(failed)}")


if __name__ == "__main__":
    main()
