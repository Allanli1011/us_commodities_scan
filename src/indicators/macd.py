"""Strict MACD triple-divergence detector — supports both top (bearish) and
bottom (bullish) divergences.

Top divergence (short):
  R1. Three swing highs strictly higher: p1 < p2 < p3
  R2. Three golden-cross DIF values strictly decreasing
  R3. Both pullbacks approach 0 from above without breaking it
  R4. Three red-bar areas strictly decreasing, each leg ≥ min_area_reduction
  R5. Third push extreme occurred within `recency_bars` of the latest bar

Bottom divergence (long) — mirror.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..config import load_config

Direction = Literal["top", "bottom"]


@dataclass
class Wave:
    direction: Literal["up", "down"]
    start_cross_idx: int
    end_cross_idx: int
    cross_value: float
    extreme_idx: int
    extreme_price: float
    hist_area: float

    def __repr__(self) -> str:
        return (
            f"Wave({self.direction}, cross@{self.start_cross_idx}={self.cross_value:+.4f}, "
            f"ext@{self.extreme_idx}={self.extreme_price:.2f}, area={self.hist_area:.2f})"
        )


UpWave = Wave  # backwards-compat alias


@dataclass(frozen=True)
class RuleCheck:
    """Result of a single MACD divergence rule check."""
    code: str           # R1..R5
    name: str           # rule name
    passed: bool
    detail: str = ""

    def label(self) -> str:
        mark = "✓" if self.passed else "✗"
        return f"{mark} {self.code} {self.name}"

    def full(self) -> str:
        s = self.label()
        return f"{s}: {self.detail}" if self.detail else s


HitKind = Literal["strict", "loose", "miss"]


@dataclass
class DivergenceResult:
    hit: bool                                     # True only on strict (5/5) match
    hit_kind: HitKind = "miss"                    # strict | loose (4/5) | miss
    direction: Direction = "top"
    waves: list[Wave] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    rule_checks: list[RuleCheck] = field(default_factory=list)
    strength: float = 0.0

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.rule_checks if c.passed)

    @property
    def n_total(self) -> int:
        return len(self.rule_checks)

    @property
    def failed_rules(self) -> list[RuleCheck]:
        return [c for c in self.rule_checks if not c.passed]

    def summary(self) -> str:
        if self.hit_kind == "miss":
            return f"NO HIT ({self.direction}) — " + (
                "; ".join(self.reasons) if self.reasons else "unknown"
            )
        kind_word = "STRICT" if self.hit_kind == "strict" else "LOOSE"
        score = f"{self.n_passed}/{self.n_total}"
        if self.waves:
            peaks = " → ".join(f"{w.extreme_price:.2f}" for w in self.waves)
            crosses = " → ".join(f"{w.cross_value:+.3f}" for w in self.waves)
            areas = " → ".join(f"{w.hist_area:.1f}" for w in self.waves)
            return (
                f"{kind_word} {self.direction} [{score}] (strength {self.strength:.2f}) | "
                f"extremes {peaks} | crossovers {crosses} | areas {areas}"
            )
        return f"{kind_word} {self.direction} [{score}]"


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Standard MACD; histogram uses the common (dif - dea) × 2 convention."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist}, index=close.index)


def find_crossovers(dif: pd.Series, dea: pd.Series) -> pd.DataFrame:
    delta = dif - dea
    prev_delta = delta.shift(1)
    up_cross = (prev_delta <= 0) & (delta > 0)
    down_cross = (prev_delta >= 0) & (delta < 0)

    records: list[dict] = []
    for pos, is_up in enumerate(up_cross.values):
        if is_up:
            records.append({"idx": pos, "type": "up", "value": float(dif.iloc[pos])})
    for pos, is_down in enumerate(down_cross.values):
        if is_down:
            records.append({"idx": pos, "type": "down", "value": float(dif.iloc[pos])})

    if not records:
        return pd.DataFrame(columns=["idx", "type", "value"]).astype(
            {"idx": int, "type": object, "value": float}
        )
    return pd.DataFrame(records).sort_values("idx").reset_index(drop=True)


def _build_waves(
    price: pd.Series,
    hist: pd.Series,
    crossovers: pd.DataFrame,
    start_type: Literal["up", "down"],
    end_type: Literal["up", "down"],
    extreme_op: Literal["max", "min"],
) -> list[Wave]:
    if crossovers.empty:
        return []

    last_bar = len(price) - 1
    starts = crossovers[crossovers["type"] == start_type]
    ends = crossovers[crossovers["type"] == end_type]
    waves: list[Wave] = []

    for _, row in starts.iterrows():
        start_idx = int(row["idx"])
        nxt = ends[ends["idx"] > start_idx]
        end_idx = int(nxt["idx"].iloc[0]) if len(nxt) else last_bar
        if end_idx < start_idx:
            continue

        seg_price = price.iloc[start_idx : end_idx + 1].values
        seg_hist = hist.iloc[start_idx : end_idx + 1].values
        if len(seg_price) == 0:
            continue

        ext_off = int(np.argmax(seg_price)) if extreme_op == "max" else int(np.argmin(seg_price))
        hist_area = float(np.abs(seg_hist).sum())

        waves.append(Wave(
            direction="up" if start_type == "up" else "down",
            start_cross_idx=start_idx,
            end_cross_idx=end_idx,
            cross_value=float(row["value"]),
            extreme_idx=start_idx + ext_off,
            extreme_price=float(seg_price[ext_off]),
            hist_area=hist_area,
        ))
    return waves


def build_up_waves(high: pd.Series, hist: pd.Series, crossovers: pd.DataFrame) -> list[Wave]:
    return _build_waves(high, hist, crossovers, "up", "down", "max")


def build_down_waves(low: pd.Series, hist: pd.Series, crossovers: pd.DataFrame) -> list[Wave]:
    return _build_waves(low, hist, crossovers, "down", "up", "min")


def check_divergence_rules(
    w1: Wave, w2: Wave, w3: Wave,
    dif: pd.Series,
    *,
    direction: Direction,
    bars_since_last_peak: int,
    min_area_reduction: float,
    dif_zero_tolerance: float,
    dif_approach_zero_ratio: float,
    min_price_increase_pct: float,
    recency_bars: int,
) -> tuple[list[RuleCheck], float]:
    """Check all 5 rules and compute pattern strength (regardless of hit)."""
    checks: list[RuleCheck] = []
    p1, p2, p3 = w1.extreme_price, w2.extreme_price, w3.extreme_price
    c1, c2, c3 = w1.cross_value, w2.cross_value, w3.cross_value
    a1, a2, a3 = w1.hist_area, w2.hist_area, w3.hist_area

    # ── R1: price makes three new extremes ──
    if direction == "top":
        inc_12 = (p2 - p1) / p1 if p1 > 0 else 0
        inc_23 = (p3 - p2) / p2 if p2 > 0 else 0
        r1_ok = inc_12 >= min_price_increase_pct and inc_23 >= min_price_increase_pct
        r1_detail = (
            f"{p1:.2f}→{p2:.2f}→{p3:.2f} (+{inc_12*100:.2f}%/+{inc_23*100:.2f}%)"
            if r1_ok else
            f"price did not make new high {p1:.2f}→{p2:.2f}→{p3:.2f} "
            f"(+{inc_12*100:.2f}%/+{inc_23*100:.2f}%)"
        )
        r1_name = "Price 3-push new high"
    else:
        dec_12 = (p1 - p2) / p1 if p1 > 0 else 0
        dec_23 = (p2 - p3) / p2 if p2 > 0 else 0
        r1_ok = dec_12 >= min_price_increase_pct and dec_23 >= min_price_increase_pct
        r1_detail = (
            f"{p1:.2f}→{p2:.2f}→{p3:.2f} (-{dec_12*100:.2f}%/-{dec_23*100:.2f}%)"
            if r1_ok else
            f"price did not make new low {p1:.2f}→{p2:.2f}→{p3:.2f} "
            f"(-{dec_12*100:.2f}%/-{dec_23*100:.2f}%)"
        )
        r1_name = "Price 3-push new low"
    checks.append(RuleCheck("R1", r1_name, r1_ok, r1_detail))

    # ── R2: DIF cross-values monotonically converge toward zero ──
    if direction == "top":
        r2_ok = c1 > c2 > c3
        r2_name = "DIF golden-cross values decreasing"
    else:
        r2_ok = c1 < c2 < c3
        r2_name = "DIF death-cross values increasing"
    r2_detail = f"{c1:+.3f}→{c2:+.3f}→{c3:+.3f}"
    checks.append(RuleCheck("R2", r2_name, r2_ok, r2_detail))

    # ── R3: DIF pullbacks approach zero without breaking it ──
    r3_problems: list[str] = []
    for k, (wA, wB) in enumerate([(w1, w2), (w2, w3)], start=1):
        seg = dif.iloc[wA.end_cross_idx : wB.start_cross_idx + 1]
        if len(seg) == 0:
            continue
        if direction == "top":
            seg_min = float(seg.min())
            if seg_min < -dif_zero_tolerance:
                r3_problems.append(f"pullback {k} broke zero (min={seg_min:.3f})")
            prev_cross = wA.cross_value
            if prev_cross > 0:
                thr = prev_cross * dif_approach_zero_ratio
                if seg_min > thr:
                    r3_problems.append(f"pullback {k} not close enough (min={seg_min:.3f} need ≤{thr:.3f})")
        else:
            seg_max = float(seg.max())
            if seg_max > dif_zero_tolerance:
                r3_problems.append(f"bounce {k} broke zero (max={seg_max:.3f})")
            prev_cross = wA.cross_value
            if prev_cross < 0:
                thr = prev_cross * dif_approach_zero_ratio
                if seg_max < thr:
                    r3_problems.append(f"bounce {k} not close enough (max={seg_max:.3f} need ≥{thr:.3f})")
    r3_ok = len(r3_problems) == 0
    r3_detail = "both pullbacks approach zero without breaking" if r3_ok else "; ".join(r3_problems)
    checks.append(RuleCheck("R3", "DIF pullbacks approach zero", r3_ok, r3_detail))

    # ── R4: histogram area strictly decays ──
    red_12 = (a1 - a2) / a1 if a1 > 0 else 0
    red_23 = (a2 - a3) / a2 if a2 > 0 else 0
    r4_monotonic = a1 > a2 > a3
    r4_enough = red_12 >= min_area_reduction and red_23 >= min_area_reduction
    r4_ok = r4_monotonic and r4_enough
    if r4_ok:
        r4_detail = f"{a1:.2f}→{a2:.2f}→{a3:.2f} decay {red_12*100:.0f}%/{red_23*100:.0f}%"
    elif not r4_monotonic:
        r4_detail = f"not monotonic {a1:.2f}/{a2:.2f}/{a3:.2f}"
    else:
        r4_detail = (
            f"decay insufficient {red_12*100:.0f}%/{red_23*100:.0f}% "
            f"(need ≥{min_area_reduction*100:.0f}%)"
        )
    checks.append(RuleCheck("R4", "Histogram area strictly decays", r4_ok, r4_detail))

    # ── R5: third push recency ──
    r5_ok = bars_since_last_peak <= recency_bars
    which = "top" if direction == "top" else "bottom"
    r5_detail = f"3rd {which} {bars_since_last_peak} bars ago (≤{recency_bars})"
    checks.append(RuleCheck("R5", "Third push recency", r5_ok, r5_detail))

    # ── strength (computed unconditionally) ──
    if direction == "top":
        price_score = min(max((p3 - p1) / max(p1, 1e-6), 0), 0.5) / 0.5
        cross_score = min(max((c1 - c3) / max(c1, 1e-6), 0), 0.9) / 0.9
    else:
        price_score = min(max((p1 - p3) / max(p1, 1e-6), 0), 0.5) / 0.5
        cross_score = min(max((c3 - c1) / max(abs(c1), 1e-6), 0), 0.9) / 0.9
    area_score = min(max((a1 - a3) / max(a1, 1e-6), 0), 0.9) / 0.9
    strength = float(np.clip((price_score + cross_score + area_score) / 3, 0, 1))
    return checks, strength


def detect_triple_divergence(
    df: pd.DataFrame,
    *,
    direction: Direction = "top",
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    min_area_reduction: float | None = None,
    dif_zero_tolerance: float | None = None,
    dif_approach_zero_ratio: float | None = None,
    min_price_increase_pct: float | None = None,
    recency_bars: int | None = None,
) -> DivergenceResult:
    cfg = load_config()["macd"]
    dcfg = cfg["divergence"]

    fast = fast if fast is not None else cfg["fast"]
    slow = slow if slow is not None else cfg["slow"]
    signal = signal if signal is not None else cfg["signal"]
    min_area_reduction = min_area_reduction if min_area_reduction is not None else dcfg["min_area_reduction"]
    dif_zero_tolerance = dif_zero_tolerance if dif_zero_tolerance is not None else dcfg["dif_zero_tolerance"]
    dif_approach_zero_ratio = dif_approach_zero_ratio if dif_approach_zero_ratio is not None else dcfg["dif_approach_zero_ratio"]
    min_price_increase_pct = min_price_increase_pct if min_price_increase_pct is not None else dcfg["min_price_increase_pct"]
    recency_bars = recency_bars if recency_bars is not None else dcfg["recency_bars"]

    needed_col = "High" if direction == "top" else "Low"
    if "Close" not in df.columns or needed_col not in df.columns:
        return DivergenceResult(
            hit=False, hit_kind="miss", direction=direction,
            reasons=[f"df missing Close or {needed_col}"],
        )
    if len(df) < slow * 2:
        return DivergenceResult(
            hit=False, hit_kind="miss", direction=direction,
            reasons=[f"only {len(df)} bars, need ≥ {slow * 2}"],
        )

    macd_df = compute_macd(df["Close"], fast, slow, signal)
    crossovers = find_crossovers(macd_df["dif"], macd_df["dea"])

    if direction == "top":
        waves = build_up_waves(df["High"], macd_df["hist"], crossovers)
    else:
        waves = build_down_waves(df["Low"], macd_df["hist"], crossovers)

    if len(waves) < 3:
        return DivergenceResult(
            hit=False, hit_kind="miss", direction=direction,
            reasons=[f"only {len(waves)} {direction}-wave(s); need ≥ 3"],
        )

    w1, w2, w3 = waves[-3:]
    bars_since_last = len(df) - 1 - w3.extreme_idx
    checks, raw_strength = check_divergence_rules(
        w1, w2, w3, macd_df["dif"],
        direction=direction,
        bars_since_last_peak=bars_since_last,
        min_area_reduction=min_area_reduction,
        dif_zero_tolerance=dif_zero_tolerance,
        dif_approach_zero_ratio=dif_approach_zero_ratio,
        min_price_increase_pct=min_price_increase_pct,
        recency_bars=recency_bars,
    )
    failed = [c for c in checks if not c.passed]
    reasons = [c.detail for c in failed]

    if len(failed) == 0:
        return DivergenceResult(
            hit=True, hit_kind="strict", direction=direction,
            waves=[w1, w2, w3], rule_checks=checks, strength=raw_strength,
        )
    if len(failed) == 1:
        # loose hit (off-by-one) — discount strength by 50% to reflect quality
        return DivergenceResult(
            hit=False, hit_kind="loose", direction=direction,
            waves=[w1, w2, w3], rule_checks=checks,
            reasons=reasons, strength=raw_strength * 0.5,
        )
    return DivergenceResult(
        hit=False, hit_kind="miss", direction=direction,
        waves=[w1, w2, w3], rule_checks=checks,
        reasons=reasons, strength=0.0,
    )


def detect_both_directions(df: pd.DataFrame, **kwargs) -> dict[Direction, DivergenceResult]:
    return {
        "top": detect_triple_divergence(df, direction="top", **kwargs),
        "bottom": detect_triple_divergence(df, direction="bottom", **kwargs),
    }


if __name__ == "__main__":
    import sys
    from ..data_fetcher import load_prices

    ticker = sys.argv[1] if len(sys.argv) > 1 else "CL=F"
    df = load_prices(ticker)
    if df is None:
        print(f"No cached prices for {ticker}. Run data_fetcher first.")
        sys.exit(1)

    results = detect_both_directions(df)
    for direction, res in results.items():
        print(f"{ticker} [{direction:>6}]: {res.summary()}")
