"""
Data loader for Munich bike counter data (Radverkehrszählstellen München).

Covers 6 stations (Arnulf, Erhardt, Hirsch, Kreuther, Margareten, Olympia)
from 2008 to present, with daily counts and weather variables.
"""

import pandas as pd
from pathlib import Path

# Canonical column names after standardisation
FINAL_COLS = [
    "date", "station", "total", "direction_1", "direction_2",
    "min_temp", "max_temp", "precipitation", "cloud_cover", "sunshine_hours",
]

# All known raw column name variants → canonical English name
_RENAME = {
    # identifiers
    "datum"        : "date",
    "zaehlstelle"  : "station",
    # count columns
    "gesamt"       : "total",
    "richtung_1"   : "direction_1",
    "richtung_2"   : "direction_2",
    # weather columns
    "niederschlag" : "precipitation",
    "bewoelkung"   : "cloud_cover",
    "sonnenstunden": "sunshine_hours",
    # temperature variants
    "min.temp"     : "min_temp",
    "max.temp"     : "max_temp",
    "mintemp"      : "min_temp",
    "maxtemp"      : "max_temp",
    "min-temp"     : "min_temp",
    "max-temp"     : "max_temp",
}


def _is_daily(filepath: Path) -> bool:
    """Return True if the file contains daily aggregates (uhrzeit_ende ≈ 23:59)."""
    try:
        sample = pd.read_csv(filepath, nrows=3)
        cols_lower = [c.lower().strip() for c in sample.columns]
        if "uhrzeit_ende" not in cols_lower:
            return False
        col = sample.columns[cols_lower.index("uhrzeit_ende")]
        end_val = str(sample[col].iloc[0]).strip()
        return "23" in end_val and "59" in end_val
    except Exception:
        return False


def _load_single(filepath: Path) -> pd.DataFrame:
    """Load one daily CSV, standardise columns, parse dates."""
    df = pd.read_csv(filepath)

    # Normalise column names to lowercase and map variants → English
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=_RENAME)

    # Parse dates: handle both YYYY-MM-DD and YYYY.MM.DD
    df["date"] = (
        df["date"]
        .astype(str)
        .str.replace(r"(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep only target columns that actually exist in this file
    present = [c for c in FINAL_COLS if c in df.columns]
    return df[present].copy()


def load_master_data(
    filepath: str | Path = "data/processed/master_bike_data.csv",
) -> pd.DataFrame:
    """
    Load the pre-built master dataset.

    Parameters
    ----------
    filepath : path to master_bike_data.csv

    Returns
    -------
    DataFrame with columns: date, station, total, direction_1, direction_2,
    min_temp, max_temp, precipitation, cloud_cover, sunshine_hours
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Master file not found: {filepath}\n"
            "Run the phase-1 notebook to generate it first."
        )
    df = pd.read_csv(filepath, parse_dates=["date"])
    df["station"] = df["station"].astype("category")
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    return df


def load_station(
    station_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    filepath: str | Path = "data/processed/master_bike_data.csv",
) -> pd.DataFrame:
    """
    Load data for a single station, optionally filtered by date range.

    Parameters
    ----------
    station_name : one of Arnulf | Erhardt | Hirsch | Kreuther | Margareten | Olympia
    start_date   : inclusive lower bound, e.g. '2015-01-01'
    end_date     : inclusive upper bound, e.g. '2022-12-31'
    filepath     : path to master_bike_data.csv

    Returns
    -------
    DataFrame filtered to the requested station and date range
    """
    df = load_master_data(filepath)

    known = df["station"].cat.categories.tolist()
    if station_name not in known:
        raise ValueError(f"Unknown station '{station_name}'. Available: {known}")

    mask = df["station"] == station_name
    if start_date:
        mask &= df["date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df["date"] <= pd.Timestamp(end_date)

    return df[mask].reset_index(drop=True)


def get_data_info(df: pd.DataFrame) -> None:
    """
    Print a concise summary of the dataset.

    Covers shape, date range, station coverage, missing values,
    and basic sanity checks (negative counts, duplicate rows).
    """
    print("=" * 60)
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Stations       : {sorted(df['station'].unique().tolist())}")
    print()

    print("Rows per station:")
    print(df.groupby("station", observed=True).size().to_string())
    print()

    print("Date range per station:")
    rng = df.groupby("station", observed=True)["date"].agg(["min", "max"])
    rng.columns = ["first", "last"]
    print(rng.to_string())
    print()

    count_cols   = ["total", "direction_1", "direction_2"]
    weather_cols = ["min_temp", "max_temp", "precipitation", "cloud_cover", "sunshine_hours"]

    print("Missing values per column:")
    missing = df[FINAL_COLS].isnull().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "  None")
    print()

    print("Negative count values:")
    neg = {c: (df[c] < 0).sum() for c in count_cols if c in df.columns}
    neg = {k: v for k, v in neg.items() if v > 0}
    print(neg if neg else "  None")

    dups = df.duplicated(subset=["date", "station"]).sum()
    print(f"\nDuplicate (date, station) rows: {dups}")
    print("=" * 60)


def build_master(
    raw_dir: str | Path = "data/raw/rad",
    output_path: str | Path = "data/processed/master_bike_data.csv",
) -> pd.DataFrame:
    """
    Scan raw_dir for daily-aggregate CSV files, merge them, and save.

    This is called by the phase-1 notebook. Re-running it overwrites
    the existing master file.
    """
    raw_dir = Path(raw_dir)
    all_files = sorted(raw_dir.glob("*.csv"))

    frames, skipped = [], []
    for f in all_files:
        if _is_daily(f):
            frames.append(_load_single(f))
        else:
            skipped.append(f.name)

    if not frames:
        raise RuntimeError(f"No daily-aggregate files found in {raw_dir}")

    master = pd.concat(frames, ignore_index=True)
    master = master.drop_duplicates(subset=["date", "station"])
    master = master.sort_values(["station", "date"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)

    print(f"Loaded {len(frames)} daily files, skipped {len(skipped)} 15-min files.")
    print(f"Master dataset: {master.shape[0]:,} rows saved to {output_path}")
    return master
