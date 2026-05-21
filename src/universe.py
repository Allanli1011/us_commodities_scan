"""Yahoo Finance futures universe.

Curated list of continuous front-month futures (`=F` suffix on Yahoo Finance)
covering energy / metals / grains / softs / livestock / index / currency /
interest-rate / crypto. The list mirrors the symbols available via yfinance
quote endpoints; tickers that Yahoo retires or relabels can simply be deleted
from `FUTURES_UNIVERSE` below.

Public API:
    build_universe(force_refresh=False) -> pd.DataFrame
    load_universe()                     -> pd.DataFrame
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from .config import load_config, project_path

logger = logging.getLogger(__name__)


# ──────────────────── Futures master list ────────────────────
# (ticker, name, category, currency, tick_size, contract_size, exchange)
# tick_size & contract_size are for reference only — sizing/PnL is not
# computed in this scanner, but we keep them so the metadata file is useful.

FUTURES_UNIVERSE: list[dict] = [
    # ── Energy ──
    {"ticker": "CL=F",  "name": "Crude Oil WTI",        "category": "Energy",   "exchange": "NYMEX"},
    {"ticker": "BZ=F",  "name": "Brent Crude Oil",      "category": "Energy",   "exchange": "ICE"},
    {"ticker": "NG=F",  "name": "Natural Gas",          "category": "Energy",   "exchange": "NYMEX"},
    {"ticker": "HO=F",  "name": "Heating Oil",          "category": "Energy",   "exchange": "NYMEX"},
    {"ticker": "RB=F",  "name": "RBOB Gasoline",        "category": "Energy",   "exchange": "NYMEX"},
    {"ticker": "MCL=F", "name": "Micro Crude Oil WTI",  "category": "Energy",   "exchange": "NYMEX"},

    # ── Metals ──
    {"ticker": "GC=F",  "name": "Gold",                 "category": "Metals",   "exchange": "COMEX"},
    {"ticker": "SI=F",  "name": "Silver",               "category": "Metals",   "exchange": "COMEX"},
    {"ticker": "HG=F",  "name": "Copper",               "category": "Metals",   "exchange": "COMEX"},
    {"ticker": "PL=F",  "name": "Platinum",             "category": "Metals",   "exchange": "NYMEX"},
    {"ticker": "PA=F",  "name": "Palladium",            "category": "Metals",   "exchange": "NYMEX"},
    {"ticker": "ALI=F", "name": "Aluminum",             "category": "Metals",   "exchange": "COMEX"},
    {"ticker": "MGC=F", "name": "Micro Gold",           "category": "Metals",   "exchange": "COMEX"},
    {"ticker": "SIL=F", "name": "Micro Silver",         "category": "Metals",   "exchange": "COMEX"},

    # ── Grains / Oilseeds ──
    {"ticker": "ZC=F",  "name": "Corn",                 "category": "Grains",   "exchange": "CBOT"},
    {"ticker": "ZS=F",  "name": "Soybeans",             "category": "Grains",   "exchange": "CBOT"},
    {"ticker": "ZW=F",  "name": "Chicago SRW Wheat",    "category": "Grains",   "exchange": "CBOT"},
    {"ticker": "KE=F",  "name": "KC HRW Wheat",         "category": "Grains",   "exchange": "CBOT"},
    {"ticker": "ZL=F",  "name": "Soybean Oil",          "category": "Grains",   "exchange": "CBOT"},
    {"ticker": "ZM=F",  "name": "Soybean Meal",         "category": "Grains",   "exchange": "CBOT"},
    {"ticker": "ZO=F",  "name": "Oats",                 "category": "Grains",   "exchange": "CBOT"},
    {"ticker": "ZR=F",  "name": "Rough Rice",           "category": "Grains",   "exchange": "CBOT"},

    # ── Softs ──
    {"ticker": "KC=F",  "name": "Coffee",               "category": "Softs",    "exchange": "ICE"},
    {"ticker": "SB=F",  "name": "Sugar #11",            "category": "Softs",    "exchange": "ICE"},
    {"ticker": "CC=F",  "name": "Cocoa",                "category": "Softs",    "exchange": "ICE"},
    {"ticker": "CT=F",  "name": "Cotton #2",            "category": "Softs",    "exchange": "ICE"},
    {"ticker": "OJ=F",  "name": "Orange Juice",         "category": "Softs",    "exchange": "ICE"},
    {"ticker": "LBR=F", "name": "Lumber",               "category": "Softs",    "exchange": "CME"},

    # ── Livestock ──
    {"ticker": "LE=F",  "name": "Live Cattle",          "category": "Livestock", "exchange": "CME"},
    {"ticker": "HE=F",  "name": "Lean Hogs",            "category": "Livestock", "exchange": "CME"},
    {"ticker": "GF=F",  "name": "Feeder Cattle",        "category": "Livestock", "exchange": "CME"},

    # ── Equity Index ──
    {"ticker": "ES=F",  "name": "E-mini S&P 500",       "category": "Index",    "exchange": "CME"},
    {"ticker": "NQ=F",  "name": "E-mini Nasdaq 100",    "category": "Index",    "exchange": "CME"},
    {"ticker": "YM=F",  "name": "E-mini Dow",           "category": "Index",    "exchange": "CBOT"},
    {"ticker": "RTY=F", "name": "E-mini Russell 2000",  "category": "Index",    "exchange": "CME"},
    {"ticker": "NKD=F", "name": "Nikkei 225 USD",       "category": "Index",    "exchange": "CME"},
    {"ticker": "MES=F", "name": "Micro E-mini S&P 500", "category": "Index",    "exchange": "CME"},
    {"ticker": "MNQ=F", "name": "Micro E-mini Nasdaq 100", "category": "Index", "exchange": "CME"},
    {"ticker": "M2K=F", "name": "Micro E-mini Russell 2000","category": "Index","exchange": "CME"},
    {"ticker": "MYM=F", "name": "Micro E-mini Dow",     "category": "Index",    "exchange": "CBOT"},

    # ── Currency ──
    {"ticker": "6E=F",  "name": "Euro FX",              "category": "Currency", "exchange": "CME"},
    {"ticker": "6B=F",  "name": "British Pound",        "category": "Currency", "exchange": "CME"},
    {"ticker": "6J=F",  "name": "Japanese Yen",         "category": "Currency", "exchange": "CME"},
    {"ticker": "6C=F",  "name": "Canadian Dollar",      "category": "Currency", "exchange": "CME"},
    {"ticker": "6A=F",  "name": "Australian Dollar",    "category": "Currency", "exchange": "CME"},
    {"ticker": "6S=F",  "name": "Swiss Franc",          "category": "Currency", "exchange": "CME"},
    {"ticker": "6N=F",  "name": "New Zealand Dollar",   "category": "Currency", "exchange": "CME"},
    {"ticker": "6M=F",  "name": "Mexican Peso",         "category": "Currency", "exchange": "CME"},
    # DX=F currently 404s on Yahoo Finance; use DX-Y.NYB index proxy or UUP
    # ETF if you need US Dollar Index exposure. Re-add when Yahoo restores it.

    # ── Interest Rate ──
    {"ticker": "ZT=F",  "name": "2-Year T-Note",        "category": "Rates",    "exchange": "CBOT"},
    {"ticker": "ZF=F",  "name": "5-Year T-Note",        "category": "Rates",    "exchange": "CBOT"},
    {"ticker": "ZN=F",  "name": "10-Year T-Note",       "category": "Rates",    "exchange": "CBOT"},
    {"ticker": "TN=F",  "name": "Ultra 10-Year T-Note", "category": "Rates",    "exchange": "CBOT"},
    {"ticker": "ZB=F",  "name": "30-Year T-Bond",       "category": "Rates",    "exchange": "CBOT"},
    {"ticker": "UB=F",  "name": "Ultra T-Bond",         "category": "Rates",    "exchange": "CBOT"},

    # ── Volatility ──
    # VX=F (continuous VIX future) is not available as a downloadable series
    # on Yahoo Finance — use ^VIX (the index) or VXX (ETN) for vol exposure.

    # ── Crypto ──
    {"ticker": "BTC=F", "name": "Bitcoin Futures",      "category": "Crypto",   "exchange": "CME"},
    {"ticker": "ETH=F", "name": "Ether Futures",        "category": "Crypto",   "exchange": "CME"},
    {"ticker": "MBT=F", "name": "Micro Bitcoin Futures", "category": "Crypto",  "exchange": "CME"},
    {"ticker": "MET=F", "name": "Micro Ether Futures",  "category": "Crypto",   "exchange": "CME"},
]


def _yahoo_safe_filename(ticker: str) -> str:
    """`CL=F` → `CL_F`, `6E=F` → `6E_F` — used as parquet filename stem."""
    return ticker.replace("=", "_").replace("^", "")


def build_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Materialise the futures universe and write to data/universe.csv."""
    cfg = load_config()
    cache_path = project_path(cfg["universe"]["cache_path"])
    refresh_days = cfg["universe"].get("refresh_days", 30)
    exclude_categories = set(cfg["universe"].get("exclude_categories", []))
    only_categories = set(cfg["universe"].get("only_categories", []))

    if not force_refresh and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            if "fetched_at" in cached.columns and len(cached) > 0:
                fetched_at = pd.to_datetime(cached["fetched_at"].iloc[0], utc=True)
                age_days = (datetime.now(timezone.utc) - fetched_at).days
                if age_days < refresh_days:
                    logger.info(
                        "Universe cache fresh (%d days old, %d tickers); skipping refresh",
                        age_days, len(cached),
                    )
                    return cached
        except Exception as e:
            logger.warning("Cache read failed (%s); rebuilding", e)

    df = pd.DataFrame(FUTURES_UNIVERSE)
    df["file_stem"] = df["ticker"].map(_yahoo_safe_filename)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()

    if only_categories:
        df = df[df["category"].isin(only_categories)].copy()
    if exclude_categories:
        df = df[~df["category"].isin(exclude_categories)].copy()
    df = df.reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, encoding="utf-8")
    logger.info("Universe written → %s (%d futures)", cache_path, len(df))
    return df


def load_universe() -> pd.DataFrame:
    cache_path = project_path(load_config()["universe"]["cache_path"])
    if not cache_path.exists():
        return build_universe()
    return pd.read_csv(cache_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_universe(force_refresh=True)
    print(df.to_string(index=False))
    print(f"\nTotal: {len(df)} futures")
    print("\nBy category:")
    print(df["category"].value_counts().to_string())
