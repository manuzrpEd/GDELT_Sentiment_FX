import yfinance as yf
import pandas as pd
from src.utils import CCYS

def get_fx_prices(start: str, end: str) -> pd.DataFrame:
    """
    Download daily FX rates for all 15 EM currencies.
    Handles missing tickers, MultiIndex hell, and weekend gaps.
    Returns clean DataFrame with currencies as columns, Date as index.
    """
    pairs = [f"{c}=X" for c in CCYS]
    
    print(f"Downloading FX prices for {len(pairs)} pairs: {', '.join(CCYS)} from {start} to {end}...")
    
    # Download with auto_adjust=True to avoid MultiIndex nightmare
    data = yf.download(
        tickers=pairs,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,      # ← THIS FIXES THE 'Adj Close' KeyError
        actions=False,
        threads=True           # ← faster
    )
    
    # If only one ticker → yf returns normal columns, not MultiIndex
    if len(pairs) == 1 or data.columns.nlevels == 1:
        price_data = data['Close' if 'Close' in data.columns else data.columns[0]]
    else:
        # MultiIndex case → safely extract Close (which is auto-adjusted)
        try:
            price_data = data['Close']
        except KeyError:
            # Fallback: some tickers failed → take whatever level exists
            price_data = data.xs('Close', axis=1, level=1, drop_level=False)
            price_data = price_data.xs('Close', axis=1, level=0)
    
    # Clean column names
    if price_data.columns.nlevels > 1:
        price_data.columns = [col[1] if isinstance(col, tuple) else col for col in price_data.columns]
    price_data.columns = [col.replace("=X", "") for col in price_data.columns]
    
    # Forward fill weekends/holidays, then drop any currency with >50% NaN
    price_data = price_data.ffill().bfill()
    
    missing = price_data.isna().mean()
    if (missing > 0.5).any():
        bad = missing[missing > 0.5].index.tolist()
        print(f"Warning: Dropping currencies with >50% missing data: {bad}")
        price_data = price_data.drop(columns=bad)
    
    print(f"Success: FX data shape: {price_data.shape} | Currencies: {', '.join(price_data.columns)}")
    return price_data

def get_fx_returns(start: str, end: str) -> pd.DataFrame:
    prices = get_fx_prices(start, end)
    returns = prices.pct_change().add_suffix("_ret")
    returns.index = returns.index.date
    return returns, prices