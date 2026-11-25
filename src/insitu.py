"""Clean the NYC Harbor Water Quality in-situ chlorophyll-a data.
Run from the root with python -m src.insitu """

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = REPOSITORY_ROOT / "data" / "raw" / "harbor_water_quality.csv"
DEFAULT_PROCESSED_DIR = REPOSITORY_ROOT / "data" / "processed"

STATION_COLUMN = "Sampling Location"
DATE_COLUMN = "Sample Date"
TIME_COLUMN = "Sample Time"
RAW_LONGITUDE_COLUMN = "Long"
RAW_LATITUDE_COLUMN = "Lat"
EXCLUDED_STATIONS = {"ert blk"}  # only one sample collected here

# Approximate bounds for the NYC Harbor study area. These are used to
# distinguish valid NYC coordinates from swapped or out-of-area coordinates.
NYC_LAT_BOUNDS = (40.0, 41.5)
NYC_LON_BOUNDS = (-74.5, -73.5)

class FileSourceError(FileNotFoundError):
    """Raised when a requested raw in-situ data file is missing."""


def find_chlorophyll_column(columns: Iterable[str]) -> str:
    """Find the top active chlorophyll-a column without relying on its units."""

    matches = [
        column
        for column in columns
        if "active chlorophyll" in str(column).lower()
        and str(column).strip().lower().startswith("top")
    ]
    if not matches:
        raise KeyError("Could not find a 'Top Active Chlorophyll' column in the input data.")
    if len(matches) > 1:
        raise ValueError(f"Found multiple possible chlorophyll columns: {matches}")
    return matches[0]


def _to_numeric(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")

def _is_nyc_pair(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Return True where a coordinate pair falls within the study area bounds"""
    return (lat.between(*NYC_LAT_BOUNDS, inclusive="both")
            & lon.between(*NYC_LON_BOUNDS, inclusive="both"))


def _normalize_coordinates(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return corrected latitude, longitude, and a flag for swapped rows.

    A pair is swapped only if the original pair is outside the NYC bounds and
    the reversed pair is inside them. Ambiguous/out-of-area pairs become NaN.
    """

    raw_lat = _to_numeric(frame[RAW_LATITUDE_COLUMN])
    raw_lon = _to_numeric(frame[RAW_LONGITUDE_COLUMN])

    original_valid = _is_nyc_pair(raw_lat, raw_lon)
    swapped_valid = _is_nyc_pair(raw_lon, raw_lat)
    swapped = ~original_valid & swapped_valid

    lat = raw_lat.where(original_valid)
    lon = raw_lon.where(original_valid)

    lat.loc[swapped] = raw_lon.loc[swapped]
    lon.loc[swapped] = raw_lat.loc[swapped]

    return lat, lon, swapped


def load_insitu(raw_path: str | Path = DEFAULT_RAW_PATH, *, 
                start_date: str | pd.Timestamp = "2016-01-01",
                end_date: str | pd.Timestamp = "2025-12-31",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and clean observations, returning rows and cleaning diagnostics.

    The returned observation-level table has one row per source observation.
    Use :func:`build_station_date_table` to aggregate it to one row per
    station and sampling date.
    """

    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileSourceError(f"In-situ data file not found: {raw_path}")

    frame = pd.read_csv(raw_path, low_memory=False)
    required = {STATION_COLUMN, DATE_COLUMN, RAW_LONGITUDE_COLUMN, RAW_LATITUDE_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    chl_column = find_chlorophyll_column(frame.columns)
    
    columns = [
            STATION_COLUMN,
            DATE_COLUMN,
            RAW_LONGITUDE_COLUMN,
            RAW_LATITUDE_COLUMN,
            chl_column,
        ]
    if TIME_COLUMN in frame.columns:
        columns.append(TIME_COLUMN)

    selected = frame[columns].copy()
    selected = selected.rename(
        columns={STATION_COLUMN: "station", chl_column: "chl"}
    )
    selected["station"] = selected["station"].astype("string").str.strip()
    selected["date"] = pd.to_datetime(
        selected[DATE_COLUMN], errors="coerce"
    ) # .dt.normalize() to set all dates to midnight of that day
    selected["chl"] = _to_numeric(selected["chl"])

    lat, lon, swapped = _normalize_coordinates(selected)
    selected["lat"] = lat
    selected["lon"] = lon

    # Keep the date as the primary matchup date. Retain a datetime only when
    # a usable sample time was supplied.
    if TIME_COLUMN in selected.columns:
        parsed_time = pd.to_datetime(
            selected[TIME_COLUMN].astype("string").str.strip(),
            format="mixed",
            errors="coerce",   
        )
        selected["datetime"] = (selected["date"] 
                                + pd.to_timedelta(parsed_time.dt.hour, unit="h")
                                + pd.to_timedelta(parsed_time.dt.minute, unit="m")
        )
    else:
        selected["datetime"] = selected["date"]

    start = pd.Timestamp(start_date)   # .normalize() to set to midnight
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date.")

    diagnostics = {
        "rows_read": len(selected),
        "coordinate_pairs_swapped": int(swapped.sum())
    }

    in_period = selected["date"].between(start, end, inclusive="both")
    diagnostics["rows_outside_date_range"] = int((~in_period).sum())
    selected = selected[in_period].copy()

    excluded = selected["station"].str.casefold().isin(EXCLUDED_STATIONS)
    diagnostics["rows_from_excluded_stations"] = int(excluded.sum())
    selected = selected[~excluded].copy()

    valid_required = (
        selected["station"].notna()
        & selected["station"].ne("")
        & selected["date"].notna()
        & selected["chl"].notna()
        
    )
    diagnostics["rows_missing_station_date_or_chl"] = int((~valid_required).sum())
    selected = selected[valid_required].copy()

    # Use each station's median position for all its rows. Recent years (2023+)
    # have no coordinates in the raw data, and some rows have wrong positions;
    # dropping them would lose about a third of the samples.
    diagnostics["rows_with_missing_coordinates_filled"] = int(
        (selected["lat"].isna() | selected["lon"].isna()).sum()
    )
    selected["lat"] = selected.groupby("station")["lat"].transform("median")
    selected["lon"] = selected.groupby("station")["lon"].transform("median")
    

    valid_coordinates = selected["lat"].notna() & selected["lon"].notna()
    diagnostics["rows_with_invalid_coordinates"] = int((~valid_coordinates).sum())
    selected = selected[valid_coordinates].copy()

    # Remove exact duplicate observations, but do not collapse same-day
    # measurements here; those are aggregated in build_station_date_table().
    before = len(selected)
    selected = selected.drop_duplicates(
        subset=["station", "date", "datetime", "chl", "lat", "lon"]
    )
    diagnostics["exact_duplicate_rows_removed"] = before - len(selected)

    output_columns = ["station", "date", "datetime", "chl", "lat", "lon"]
    selected = selected[output_columns].sort_values(
        ["station", "datetime"], kind="stable"
    ).reset_index(drop=True)

    diagnostics["clean_observation_rows"] = len(selected)
    diagnostics["station_count"] = int(selected["station"].nunique())

    return selected, diagnostics


def clean_insitu(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    *,
    start_date: str | pd.Timestamp = "2016-01-01",
) -> pd.DataFrame:
    """Compatibility name for :func:`load_insitu`."""

    return load_insitu(raw_path, start_date=start_date)


def build_station_date_table(insitu: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observations to one row per station and sampling date.
    Might be useful later."""

    daily = (
        insitu.groupby(["station", "date"], as_index=False)
        .agg(
            chl=("chl", "mean"),
            sample_count=("chl", "size"),
            lat=("lat", "median"),
            lon=("lon", "median"),
        )
        .sort_values(["station", "date"])
        .reset_index(drop=True)
    )

    counts = (insitu.groupby(["station", "date"])
              .size()
              .sort_values(ascending=False)
    )
    print("Station-date groups with multiple rows:", (counts > 1).sum())

    return daily

def build_station_table(insitu: pd.DataFrame) -> pd.DataFrame:
    """Summarize station coordinates and sample coverage."""
    stations = (
        insitu.groupby("station", as_index=False)
        .agg(
            lat=("lat", "median"),
            lon=("lon", "median"),
            n_observations=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values("station")
        .reset_index(drop=True)
    )

    # Flag shared coordinates for review. Do not automatically remove stations:
    # two different stations may legitimately have the same recorded location.
    stations["shared_coordinate"] = stations.duplicated(subset=["lat", "lon"], keep=False)

    return stations

def save_outputs(
    observations: pd.DataFrame,
    output_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> tuple[Path, Path, Path]:
    """Save cleaned observations, station-date rows, and station summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations_path = output_dir / "insitu_observations.csv"
    station_date_path = output_dir / "insitu_station_date.csv"
    stations_path = output_dir / "stations.csv"


    # was: observations.to_csv(observations_path, index=False, date_format="%Y-%m-%d")
    observations.to_csv(observations_path, index=False)

    # Might be useful later. Current data has one observation per station per date
    build_station_date_table(observations).to_csv(
        station_date_path, index=False, date_format="%Y-%m-%d"
    )
    build_station_table(observations).to_csv(
        stations_path, index=False, date_format="%Y-%m-%d"
    )
    return observations_path, station_date_path, stations_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean NYC Harbor in-situ chlorophyll-a observations."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args(argv)

    observations, diagnostics = load_insitu(args.raw, start_date=args.start_date, end_date=args.end_date)
    paths= save_outputs(observations, args.output_dir)

    for label, count in diagnostics.items():
        print(f"{label}: {count:,}")   # comma as thousands separator
    for path in paths:
        print(f"Saved: {path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())