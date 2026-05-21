"""Full futures scan.

Flow:
  1. Load (or refresh) futures universe + incrementally update daily bars
  2. For every contract, run MACD triple divergence / 3-push / HTF PDA
  3. Composite score (max 3.0), sort by score then category, write CSV
  4. Optionally render annotated K-line charts for the top N

Usage:
  python scripts/scan_full.py --min-score 1.5 --plot-top 5
  python scripts/scan_full.py --direction bottom --refresh-data
  python scripts/scan_full.py --category Energy --plot-top 0
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, project_path
from src.data_fetcher import load_prices, refresh_all, update_prices
from src.indicators.macd import DivergenceResult, detect_triple_divergence as detect_macd
from src.indicators.pda import PDAResult, detect_htf_pda_hit
from src.indicators.swing import find_swing_points
from src.indicators.three_push import ThreePushResult, detect_three_push
from src.universe import _yahoo_safe_filename, load_universe

logger = logging.getLogger(__name__)
Direction = Literal["top", "bottom"]


def _stop_buffer_pct() -> float:
    return 0.01


def _currency() -> str:
    return load_config().get("runtime", {}).get("currency_symbol", "$")


def build_notes(
    direction: Direction,
    macd_res: DivergenceResult,
    tp_res: ThreePushResult,
    pda_res: PDAResult,
    target_price: float,
) -> str:
    cur = _currency()
    parts: list[str] = []
    label = "top" if direction == "top" else "bottom"

    if tp_res.hit and tp_res.pullbacks:
        e1, e2, e3 = tp_res.extremes
        p1, p2 = tp_res.pullbacks
        pull_word = "pullback" if direction == "top" else "bounce"
        parts.append(
            f"3-push {label} {e1.price:.2f}→{e2.price:.2f}→{e3.price:.2f} "
            f"{pull_word} {p1*100:.0f}%/{p2*100:.0f}%"
        )

    if macd_res.hit_kind == "strict":
        parts.append(
            f"MACD STRICT {label} div ({macd_res.n_passed}/{macd_res.n_total}, "
            f"strength {macd_res.strength:.2f})"
        )
    elif macd_res.hit_kind == "loose":
        failed = macd_res.failed_rules
        miss_word = failed[0].code if failed else ""
        parts.append(
            f"MACD LOOSE {label} div ({macd_res.n_passed}/{macd_res.n_total}, "
            f"miss {miss_word}, strength {macd_res.strength:.2f})"
        )

    if pda_res.hit:
        first = pda_res.hits[0]
        tf_word = "Weekly" if first.timeframe == "W" else "Monthly"
        parts.append(
            f"{tf_word} {pda_res.best_quality} [{first.zone.zone_low:.2f}-{first.zone.zone_high:.2f}]"
        )

    if pda_res.hit and (tp_res.hit or macd_res.hit_kind != "miss"):
        zone = pda_res.hits[0].zone
        origin_price = tp_res.origin.price if (tp_res.hit and tp_res.origin) else None
        buffer = _stop_buffer_pct()

        if direction == "bottom":
            stop = zone.zone_low * (1 - buffer)
            entry = target_price
            target = origin_price
            if target is not None and target > entry > stop:
                rr = (target - entry) / (entry - stop)
                parts.append(
                    f"LONG: entry {cur}{entry:.2f} stop {cur}{stop:.2f} "
                    f"target {cur}{target:.2f} R:R {rr:.1f}"
                )
            elif target is None:
                parts.append(
                    f"LONG: entry {cur}{entry:.2f} stop {cur}{stop:.2f} "
                    f"(no clear target, PDA-only)"
                )
        else:
            stop = zone.zone_high * (1 + buffer)
            entry = target_price
            target = origin_price
            if target is not None and stop > entry > target:
                rr = (entry - target) / (stop - entry)
                parts.append(
                    f"SHORT: entry {cur}{entry:.2f} stop {cur}{stop:.2f} "
                    f"target {cur}{target:.2f} R:R {rr:.1f}"
                )
            elif target is None:
                parts.append(
                    f"SHORT: entry {cur}{entry:.2f} stop {cur}{stop:.2f} "
                    f"(no clear target, PDA-only)"
                )

    return " | ".join(parts) if parts else "(no significant signal)"


def _list_cached_tickers() -> list[str]:
    """List tickers that have a cached parquet. We map file-stem back to ticker
    by consulting the universe metadata; orphan files are dropped."""
    universe = load_universe()
    available = {row["file_stem"]: row["ticker"] for _, row in universe.iterrows()}
    stems = sorted(f.stem for f in project_path("data/prices").glob("*.parquet"))
    return [available[s] for s in stems if s in available]


def _load_metadata() -> tuple[dict[str, str], dict[str, str]]:
    """name + category lookup keyed by ticker."""
    universe = load_universe()
    names = dict(zip(universe["ticker"], universe["name"]))
    cats = dict(zip(universe["ticker"], universe["category"]))
    return names, cats


def _target_price_for(df: pd.DataFrame, direction: Direction) -> tuple[float, int]:
    swings = find_swing_points(df)
    target_kind = "high" if direction == "top" else "low"
    candidates = [s for s in swings if s.kind == target_kind]
    if candidates:
        last = candidates[-1]
        return last.price, last.idx
    return float(df["Close"].iloc[-1]), len(df) - 1


def _pda_score(quality: str) -> float:
    if quality == "OB+FVG":
        return 1.0
    if quality in ("OB", "FVG"):
        return 0.5
    return 0.0


def _macd_score(kind: str) -> float:
    """strict +1.0, loose +0.5, miss 0."""
    return {"strict": 1.0, "loose": 0.5}.get(kind, 0.0)


def scan_one(
    ticker: str, df: pd.DataFrame,
    name: str, category: str, direction: Direction,
) -> dict:
    target_price, target_idx = _target_price_for(df, direction)
    target_date = df.index[target_idx].date()

    macd_res = detect_macd(df, direction=direction)
    tp_res = detect_three_push(df, direction=direction)
    pda_res = detect_htf_pda_hit(df, target_price=target_price, direction=direction)

    score = 0.0
    score += _macd_score(macd_res.hit_kind)
    score += 1.0 if tp_res.hit else 0.0
    score += _pda_score(pda_res.best_quality)

    failed_codes = ",".join(c.code for c in macd_res.failed_rules)
    row: dict = {
        "ticker": ticker,
        "name": name,
        "category": category,
        "direction": direction,
        "signal": "SHORT" if direction == "top" else "LONG",
        "score": round(score, 2),
        "last_close": round(float(df["Close"].iloc[-1]), 4),
        "last_date": df.index[-1].date(),
        "target_price": round(target_price, 4),
        "target_date": target_date,
        "macd_kind": macd_res.hit_kind,
        "macd_passed": f"{macd_res.n_passed}/{macd_res.n_total}" if macd_res.n_total else None,
        "macd_failed_rules": failed_codes or None,
        "macd_strength": round(macd_res.strength, 3) if macd_res.hit_kind != "miss" else None,
        "three_push_hit": tp_res.hit,
        "three_push_quality": round(tp_res.quality, 3) if tp_res.hit else None,
        "pda_hit": pda_res.hit,
        "pda_quality": pda_res.best_quality if pda_res.hit else None,
    }
    if pda_res.hit:
        first = pda_res.hits[0]
        row.update({
            "pda_timeframe": first.timeframe,
            "pda_zone_low": round(first.zone.zone_low, 4),
            "pda_zone_high": round(first.zone.zone_high, 4),
        })

    row["notes"] = build_notes(direction, macd_res, tp_res, pda_res, target_price)
    return row


def scan_all(
    direction: Literal["both", "top", "bottom"],
    min_score: float,
    category_filter: str | None = None,
) -> pd.DataFrame:
    tickers = _list_cached_tickers()
    names, cats = _load_metadata()
    directions: list[Direction] = ["top", "bottom"] if direction == "both" else [direction]

    if category_filter:
        tickers = [t for t in tickers if cats.get(t) == category_filter]

    rows: list[dict] = []
    for t in tqdm(tickers, desc="scanning"):
        df = load_prices(t)
        if df is None or df.empty:
            continue
        for d in directions:
            row = scan_one(t, df, names.get(t, ""), cats.get(t, ""), d)
            if row["score"] >= min_score:
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["score", "category", "ticker"], ascending=[False, True, True],
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["both", "top", "bottom"], default="both")
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="minimum composite score to include in CSV (default 1.0)")
    parser.add_argument("--plot-top", type=int, default=3,
                        help="auto-render top N signal charts (0 = none)")
    parser.add_argument("--refresh-data", action="store_true",
                        help="rebuild universe + refresh price cache before scanning")
    parser.add_argument("--update-prices", action="store_true",
                        help="incremental price update only (skip universe rebuild)")
    parser.add_argument("--category", type=str, default=None,
                        help="filter to a single category (Energy, Metals, Grains, ...)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    out_dir = project_path(cfg["output"]["csv_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_data:
        refresh_all()
    elif args.update_prices:
        from src.universe import load_universe as _lu
        update_prices(_lu()["ticker"].tolist())

    df = scan_all(args.direction, args.min_score, category_filter=args.category)
    today = date.today().isoformat()
    suffix = f"_{args.category}" if args.category else ""
    path = out_dir / f"futures_signals_{today}{suffix}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Combined signals: %d rows → %s", len(df), path)

    if len(df):
        print(f"\n── Scan results (score ≥ {args.min_score}) ──")
        show_cols = [
            "ticker", "name", "category", "direction", "signal", "score",
            "last_close", "macd_kind", "macd_passed",
            "three_push_hit", "pda_hit", "pda_quality",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        print(df[show_cols].head(25).to_string(index=False))

        print("\n── Score distribution ──")
        print(df["score"].value_counts().sort_index(ascending=False).to_string())

        triple = df[df["score"] >= 2.5]
        if len(triple):
            print(f"\n🎯 Triple convergence (score ≥ 2.5): {len(triple)}")
            print(triple[show_cols].to_string(index=False))

        if args.plot_top > 0:
            from src.visualization import render_signal_chart
            charts_dir = project_path("output/charts")
            top_rows = df.head(args.plot_top)
            print(f"\n── Rendering {len(top_rows)} chart(s) ──")
            for _, row in top_rows.iterrows():
                price_df = load_prices(row["ticker"])
                if price_df is None:
                    continue
                safe = _yahoo_safe_filename(row["ticker"])
                chart_path = charts_dir / f"{today}_{safe}_{row['direction']}.png"
                render_signal_chart(
                    ticker=row["ticker"], name=row.get("name", ""),
                    df=price_df, direction=row["direction"],
                    score=float(row["score"]), notes=row.get("notes", ""),
                    output_path=chart_path,
                )
                print(f"  → {chart_path}")
    else:
        print(f"\nNo signals at score ≥ {args.min_score}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
