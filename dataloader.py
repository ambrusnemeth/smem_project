from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

# -----------------
# Module defaults
# -----------------
DATA_DIR = Path(".")
cutoff_date = pd.Timestamp("2019-11-19")  # master reference date
product_code = "DEBY 2021.01"              # calendar 2021 baseload (per exercise)
r = 0.01                                   # kept for parity with original snippet

# -----------------
# Helpers
# -----------------
def _norm(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())

def pick(cols, candidates, required=True):
    ncols = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in cols}
    for cand in candidates:
        cand = re.sub(r'[^a-z0-9]', '', cand.lower())
        for n, orig in ncols.items():
            if cand in n:
                return orig
    if required:
        raise KeyError(f"None of {candidates} found in columns: {list(cols)}")
    return None

# -----------------
# 1) Historical FWD
# -----------------
def load_hfwd(csv_path: Path | str = DATA_DIR / "Historical_Prices_FWD_Germany.csv",
              date_limit: pd.Timestamp = cutoff_date):
    """
    Loads the historical forward prices and returns a pandas Series (index: date, values: DEBY2021),
    filtered to positive values, non-null, up to date_limit, sorted by date ascending.
    Defaults to using global cutoff_date.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Data": "date"})
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["DEBY2021"]
    s = s[(s > 0)].dropna().loc[:date_limit].sort_index()
    return s

# -----------------
# 2) Options snapshot
# -----------------
def load_options_snapshot(
    csv_path: Path | str = DATA_DIR / "Options_Prices_Calendar_2021.csv",
    trade_date_in: pd.Timestamp = cutoff_date,
    product_code_in: str = product_code,
):
    """
    Reads the options file and returns the snapshot (call options) for the given trade_date/product_code,
    with columns:
      K: strike (float)
      T: time-to-expiry in years (float)
      P: option price (float)
    Sorted by ["T", "K"] and index reset.
    Defaults to using cutoff_date as trade_date.
    """
    opts = pd.read_csv(csv_path)

    date_col   = pick(opts.columns, ["tradingdate", "pricedate", "date", "data"])
    expiry_col = pick(opts.columns, ["expirydate", "expirationdate", "maturity", "expiry"])
    under_col  = pick(opts.columns, ["underlying", "product", "symbol", "name", "contract"])
    type_col   = pick(opts.columns, ["optiontype", "type", "callput", "cp"])
    strike_col = pick(opts.columns, ["strikeprice", "strike", "k"])
    price_col  = pick(opts.columns, ["price", "settlementprice", "settlement", "mid", "last", "close", "premium"])

    opts[date_col]   = pd.to_datetime(opts[date_col], errors="coerce")
    opts[expiry_col] = pd.to_datetime(opts[expiry_col], errors="coerce")

    mask_under = opts[under_col].astype(str).map(_norm) == _norm(product_code_in)
    mask_date  = opts[date_col] == trade_date_in
    mask_type  = opts[type_col].astype(str).str.upper().str.startswith("C")

    snap = opts.loc[mask_under & mask_date & mask_type].copy()

    T = (snap[expiry_col] - trade_date_in).dt.days / 365.0
    snap = snap.assign(
        T=T.values,
        K=snap[strike_col].astype(float).values,
        P=snap[price_col].astype(float).values
    )
    snap = snap[["K", "T", "P"]].sort_values(["T", "K"]).reset_index(drop=True)
    return snap

# -----------------
# 3) Forwards
# -----------------
def load_forward_price(
    csv_path: Path | str = DATA_DIR / "Forward_Prices.csv",
    trade_date_in: pd.Timestamp = cutoff_date,
    product_code_in: str = product_code,
) -> float:
    """
    Reads the forwards file and returns F0 (float) using the exact selection logic you provided.
    Defaults to using cutoff_date as trade_date.
    """
    fwds = pd.read_csv(csv_path)

    f_date  = pick(fwds.columns, ["tradingdate", "date", "data"])
    f_name  = pick(fwds.columns, ["underlying", "product", "name", "contract"])
    f_per   = pick(fwds.columns, ["deliveryperiod", "period", "maturity"], required=False)
    f_price = pick(fwds.columns, ["price", "settlementprice", "settlement", "mid", "last", "close", "fixing"])

    fwds[f_date] = pd.to_datetime(fwds[f_date], errors="coerce")

    # prefer CONTRACT first
    f_name  = pick(fwds.columns, ["contract", "underlying", "product", "name"])

    fwds[f_date] = pd.to_datetime(fwds[f_date], errors="coerce")

    target_contract = _norm(product_code_in.split()[0])
    target_year = re.sub(r'\D', '', product_code_in.split()[1])[:4]

    m = fwds[f_date].le(trade_date_in) & fwds[f_name].astype(str).map(_norm).str.contains(target_contract)

    if f_per is not None:
        per_norm = fwds[f_per].astype(str).str.extract(r'(\d{4})', expand=False).fillna("")
        m &= per_norm.eq(target_year)

    candidates = fwds.loc[m].copy()

    if candidates.empty:
        debug = fwds.loc[fwds[f_name].astype(str).map(_norm).str.contains(target_contract)]
        raise ValueError(
            f"No forward for {target_contract} {target_year} on/before {trade_date_in.date()}.\n"
            f"Examples available:\n{debug[[f_date, f_name] + ([f_per] if f_per else [])].head(8)}"
        )

    candidates = candidates.sort_values(f_date)
    F0 = float(pd.to_numeric(candidates[f_price], errors="coerce").dropna().iloc[-1])
    return F0

__all__ = [
    "DATA_DIR", "cutoff_date", "product_code", "r",
    "load_hfwd", "load_options_snapshot", "load_forward_price",
    "_norm", "pick",
]