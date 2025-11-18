# src/data/loader.py
import pandas as pd
from pathlib import Path
from src.data.gdelt import query_gdelt_sentiment_bulk  # ← Bulk HTTP
from src.data.price import get_fx_prices

def load_full_dataset(
    start_date: str = "2018-01-01",
    end_date: str = "2025-11-17",
    cache_path: str = "data/processed/gdelt_fx_full.parquet"
) -> pd.DataFrame:
    path = Path(cache_path)
    if path.exists():
        print("Loading cached dataset...")
        return pd.read_parquet(path)
    else:
        print("Building fresh dataset via HTTP (no quota) – it will take time...")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Single bulk pull (replaces loop)
    sentiment = query_gdelt_sentiment_bulk(start_date, end_date)
    sentiment.set_index(['event_date', 'currency'], inplace=True)

    # Prices & returns (your code)
    prices = get_fx_prices(start_date, pd.Timestamp(end_date) + pd.Timedelta(days=3))
    returns = prices.pct_change().shift(-1)

    # Merge
    sentiment_wide = sentiment.pivot_table(
        index='event_date',
        columns='currency',
        values=['avg_tone', 'tone_dispersion', 'event_count'],
        aggfunc='first'  # safe because you already have one row per (date,ccy)
    )
    sentiment_wide.columns = [f"{col[0]}_{col[1]}".lower() for col in sentiment_wide.columns]
    sentiment_wide = sentiment_wide.astype('float32')  # save memory
    sentiment_wide.index = pd.to_datetime(sentiment_wide.index)
    returns.index = pd.to_datetime(returns.index).date
    sentiment_wide.index.name = 'event_date'
    returns.index.name = 'event_date'
    df = sentiment_wide.join(returns, how="inner")
    df = df.dropna()

    df.to_parquet(path, compression="zstd")
    print(f"Dataset built & cached → {path} ({path.stat().st_size // 1024 // 1024} MB)")
    return df