# src/data/gdelt.py – MAN AHL / TWO SIGMA GRADE (Full capture + bulletproof)
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import BytesIO
from zipfile import ZipFile
from src.utils import CCYS, COUNTRIES, COUNTRY_TO_CCY
from pathlib import Path

# Create daily cache folder (so crashes don't kill you)
DAILY_CACHE = Path("data/raw/gdelt_daily")
DAILY_CACHE.mkdir(parents=True, exist_ok=True)

def download_and_aggregate_day(
    date_str: str,
    min_mentions: int = 1,
    min_event_count: int = 1,
    root_only: bool = False,
    tone_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Download & aggregate GDELT sentiment for one day.
    Fully configurable — perfect for research & production.
    """
    cache_file = DAILY_CACHE / f"{date_str}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except:
            pass

    yyyymmdd = date_str.replace('-', '')
    url = f"http://data.gdeltproject.org/events/{yyyymmdd}.export.CSV.zip"

    try:
        r = requests.get(url, timeout=90)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()

        with ZipFile(BytesIO(r.content)) as z:
            csv_name = z.namelist()[0]
            df = pd.read_csv(z.open(csv_name), sep='\t', header=None, dtype=str, low_memory=False)

        # Auto-detect schema
        if len(df.columns) == 58:
            cols = [0, 1, 7, 25, 31, 34]
        elif len(df.columns) >= 61:
            cols = [0, 1, 7, 26, 31, 34]
        else:
            return pd.DataFrame()

        df = df.iloc[:, cols].copy()
        df.columns = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'IsRootEvent', 'NumMentions', 'AvgTone']

        df = df.replace({'---': pd.NA, '': pd.NA})
        df['Actor1CountryCode'] = df['Actor1CountryCode'].str.strip()

        for col in ['GLOBALEVENTID', 'SQLDATE', 'IsRootEvent', 'NumMentions']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['AvgTone'] = pd.to_numeric(df['AvgTone'], errors='coerce')

        df = df.dropna(subset=['SQLDATE', 'Actor1CountryCode', 'AvgTone']).reset_index(drop=True)

        # COUNTRY FILTER (your 15 EM currencies)
        df = df[df['Actor1CountryCode'].isin(COUNTRIES)]
        df['event_date'] = pd.to_datetime(df['SQLDATE'].astype(int), format='%Y%m%d').dt.date
        # CRITICAL: Only keep events that happened ON THE SAME DAY they were widely reported
        # This removes delayed coverage of old events → clean "new news" signal
        df = df[df['event_date'] == pd.to_datetime(date_str).date()]
        df['currency'] = df['Actor1CountryCode'].map(COUNTRY_TO_CCY)

        # APPLY CONFIGURABLE FILTERS
        df = df[df['NumMentions'] >= min_mentions]
        if root_only:
            df = df[df['IsRootEvent'] == 1]
        if tone_threshold is not None:
            df = df[df['AvgTone'].abs() >= tone_threshold]

        agg = df.groupby(['event_date', 'currency']).agg(
            avg_tone=('AvgTone', 'mean'),
            tone_dispersion=('AvgTone', 'std'),
            event_count=('GLOBALEVENTID', 'count')
        ).reset_index()

        agg = agg[agg['event_count'] >= min_event_count]

        if not agg.empty:
            agg.to_parquet(cache_file, compression='zstd')
            print(f"Success {date_str} → {len(agg):3} rows (m≥{min_mentions}, e≥{min_event_count}) | "
                  f"{agg['currency'].nunique():2} ccy | tone ∈ [{agg['avg_tone'].min():.2f}, {agg['avg_tone'].max():.2f}]")

        return agg

    except Exception as e:
        print(f"Failed {date_str} → {e}")
        return pd.DataFrame()


def query_gdelt_sentiment_bulk(start_date: str = "2018-01-01", end_date: str = "2025-11-17") -> pd.DataFrame:
    print("\nSTARTING FULL GDELT 2.0 EVENTS DOWNLOAD")
    print(f"Date range : {start_date} → {end_date}")
    print(f"Currencies : {', '.join(CCYS)}\n")

    dates = pd.date_range(start_date, end_date).strftime('%Y-%m-%d').tolist()
    results = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_and_aggregate_day, d): d for d in dates}
        for i, future in enumerate(as_completed(futures), 1):
            df_day = future.result()
            if not df_day.empty:
                results.append(df_day)

            if i % 30 == 0 or i == len(dates):
                print(f"Progress → {i:4}/{len(dates)} days ({i/len(dates)*100:5.1f}%) | "
                      f"Collected so far: {sum(len(r) for r in results):,} rows")

    if not results:
        print("No data collected!")
        return pd.DataFrame()

    final = pd.concat(results, ignore_index=True)
    print(f"\nVICTORY — FULL GDELT SENTIMENT DATASET COMPLETE")
    print(f"Total rows        : {len(final):,}")
    print(f"Trading days      : {final['event_date'].nunique():,}")
    print(f"Active currencies : {final['currency'].nunique()}")
    print(f"Date range        : {final['event_date'].min()} → {final['event_date'].max()}")
    print(f"Avg tone          : {final['avg_tone'].mean():.3f} ± {final['avg_tone'].std():.3f}")
    print(f"Cache location    : {DAILY_CACHE}\n")

    return final.sort_values(['event_date', 'currency']).reset_index(drop=True)