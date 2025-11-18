import pandas as pd
import vectorbt as vbt
from src.utils import CCYS
from typing import Tuple, List

def build_portfolio_signals(
    df: pd.DataFrame,
    pred_col: str = "pred_return",
    top_n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expects a long-format df with columns ['date','currency', pred_col]
    Returns boolean DataFrames (entries) indexed by date and columns = currency tickers (uppercase)
    """
    df = df.copy()
    # ensure date column exists (if index was date)
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or 0: "date"})

    # rank per date
    df['rank'] = df.groupby('date')[pred_col].rank(ascending=False, method='first')
    # determine top/bottom
    unique_ccys = sorted(df['currency'].unique(), key=lambda x: x.upper())
    n_ccys = len(unique_ccys)
    df['long'] = df['rank'] <= top_n
    df['short'] = df['rank'] > (n_ccys - top_n)

    long_entries = df.pivot(index='date', columns='currency', values='long').fillna(False)
    short_entries = df.pivot(index='date', columns='currency', values='short').fillna(False)

    # make columns uppercase to match price DataFrame convention
    long_entries.columns = [c.upper() for c in long_entries.columns]
    short_entries.columns = [c.upper() for c in short_entries.columns]

    # ensure consistent column ordering across outputs (use union with CCYS if available)
    if CCYS:
        cols = [c for c in CCYS if c in long_entries.columns or c in short_entries.columns]
        cols_upper = [c.upper() for c in cols]
        long_entries = long_entries.reindex(columns=cols_upper, fill_value=False)
        short_entries = short_entries.reindex(columns=cols_upper, fill_value=False)

    return long_entries, short_entries

def run_backtest(prices: pd.DataFrame, signals_long: pd.DataFrame, signals_short: pd.DataFrame):
    """
    Equal-weight long/short portfolio with 5 bps transaction cost
    """
    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=signals_long,
        short_entries=signals_short,
        freq='1D',
        fees=0.0005,  # 5 bps
        slippage=0.0001
    )
    return pf

def print_performance(pf):
    stats = pf.stats()
    print(f"Total Return: {pf.total_return():.1%}")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max Drawdown']:.1%}")
    print(f"Win Rate: {stats['Win Rate']:.1%}")
    pf.plot().show()

# New helper used by streamlit: predict on wide-format df and build signals
def build_signals(df: pd.DataFrame, model, scaler, top_n: int = 5, currencies: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accepts wide-format df with avg_tone_<ccy>, event_count_<ccy>, tone_dispersion_<ccy>.
    Returns long_sig, short_sig boolean DataFrames indexed by date and columns=UPPER currency tickers.
    """
    df2 = df.reset_index()  # keep date column
    if currencies is None:
        tone_cols = [c for c in df2.columns if c.startswith("avg_tone_")]
        currencies = sorted([c.replace("avg_tone_", "") for c in tone_cols])

    rows = []
    for c in currencies:
        avg_col = f"avg_tone_{c}"
        ev_col = f"event_count_{c}"
        disp_col = f"tone_dispersion_{c}"
        # choose return col if present
        ret_col = None
        for cand in (c.upper(), c.lower(), f"{c}_ret"):
            if cand in df2.columns:
                ret_col = cand
                break
        sub = df2[[df2.columns[0]]].copy()
        sub = sub.rename(columns={df2.columns[0]: "date"})
        sub["currency"] = c
        sub["avg_tone"] = df2[avg_col] if avg_col in df2.columns else 0
        sub["event_count"] = df2[ev_col] if ev_col in df2.columns else 0
        sub["tone_dispersion"] = df2[disp_col] if disp_col in df2.columns else 0
        if ret_col:
            sub["next_day_return"] = df2[ret_col].values
        else:
            sub["next_day_return"] = df2["next_day_return"].values if "next_day_return" in df2.columns else pd.NA
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True).dropna(subset=["next_day_return"])

    # prepare features, scale and predict
    X = long[["avg_tone", "event_count", "tone_dispersion"]].fillna(0)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    long["pred_return"] = preds

    # build signals from predicted returns
    long_entries, short_entries = build_portfolio_signals(long, pred_col="pred_return", top_n=top_n)
    return long_entries, short_entries