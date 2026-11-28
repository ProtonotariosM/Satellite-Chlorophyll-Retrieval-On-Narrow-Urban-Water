"""Build the satellite-in-situ matchup table from the Earth Engine exports.

Input: the yearly pixel-level exports from src/gee_extract.py
(data/processed/s2_pixel_matchups_<year>.csv), one row per pixel.

Output: data/processed/matchups.csv, one row per in-situ sample and
Sentinel-2 image, with the mean band values and indices of the clear
pixels in the station buffer.

Run from the root with: python -m src.matchups
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = REPOSITORY_ROOT / "data" / "processed"
PIXEL_FILE_PATTERN = "s2_pixel_matchups_*.csv"
DEFAULT_OUTPUT = DEFAULT_PROCESSED_DIR / "matchups.csv"

# Columns that Earth Engine adds to every export; not needed here
EXPORT_COLUMNS = ["system:index", ".geo"]

BANDS = ["B2", "B3", "B4", "B5", "B8", "B8A"]
INDICES = ["NDVI", "NDCI", "B8AB4", "B3B2"]

# Information about the sample and the image, identical for all pixels
# of one sample-image pair
PAIR_COLUMNS = [
    "sample_id", "station", "sample_date", "chl", "station_lat", "station_lon",
    "image_id", "image_date", "image_time_utc", "day_difference", "processing",
]

# A 15 m buffer on a 10 m grid contains about 7 pixels. Pairs with fewer
# clear pixels left after cloud masking give unreliable averages.
DEFAULT_MIN_PIXELS = 4
# The export already masks pixels with cs_cdf < 0.60; a higher value here
# removes hazy pixels without re-running the extraction.
DEFAULT_CLOUD_THRESHOLD = 0.60


def load_pixels(processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
                pattern: str = PIXEL_FILE_PATTERN) -> pd.DataFrame:
    """Read and combine all yearly pixel exports."""

    paths = sorted(Path(processed_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {pattern} in {processed_dir}. "
            "Download the Earth Engine exports from Google Drive first."
        )

    pixels = pd.concat(
        (pd.read_csv(path, low_memory=False) for path in paths), ignore_index=True
    )
    pixels = pixels.drop(columns=[c for c in EXPORT_COLUMNS if c in pixels.columns])
    pixels["sample_date"] = pd.to_datetime(pixels["sample_date"])
    pixels["image_date"] = pd.to_datetime(pixels["image_date"])
    pixels["day_difference"] = pixels["day_difference"].astype(int)
    print(f"Read {len(paths)} files: {', '.join(p.name for p in paths)}")
    return pixels


def build_matchups(
    pixels: pd.DataFrame,
    *,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    cloud_threshold: float = DEFAULT_CLOUD_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate pixels to one row per sample and image.

    Steps:
    1. Drop pixels below the cloud-score threshold.
    2. Average the remaining pixels per sample and image (the original
       paper also uses the mean index value within the 15 m buffer).
    3. Drop pairs with fewer than `min_pixels` clear pixels.
    4. Where two images of the same day exist for one sample (overlapping
       Sentinel-2 tiles), keep the one with the most clear pixels, then
       the highest cloud score.
    """

    diagnostics = {"pixel_rows_read": len(pixels)}

    clear = pixels["cs_cdf"] >= cloud_threshold
    diagnostics["pixels_below_cloud_threshold"] = int((~clear).sum())
    pixels = pixels[clear]

    value_columns = BANDS + INDICES + ["cs_cdf"]
    pairs = pixels.groupby(["sample_id", "image_id"], as_index=False).agg(
        **{c: (c, "first") for c in PAIR_COLUMNS if c not in ("sample_id", "image_id")},
        **{c: (c, "mean") for c in value_columns},
        n_pixels=("B4", "size"),
    )
    diagnostics["sample_image_pairs"] = len(pairs)

    enough = pairs["n_pixels"] >= min_pixels
    diagnostics["pairs_with_too_few_pixels"] = int((~enough).sum())
    pairs = pairs[enough]

    before = len(pairs)
    pairs = (
        pairs.sort_values(["n_pixels", "cs_cdf"], ascending=False)
        .drop_duplicates(subset=["sample_id", "image_date"], keep="first")
    )
    diagnostics["same_day_duplicate_images_removed"] = before - len(pairs)

    pairs = pairs.rename(columns={"cs_cdf": "cs_cdf_mean"})
    pairs = pairs[PAIR_COLUMNS + ["n_pixels", "cs_cdf_mean"] + BANDS + INDICES]
    pairs = pairs.sort_values(["station", "sample_date", "day_difference"]).reset_index(drop=True)

    diagnostics["matchup_rows"] = len(pairs)
    for window in (0, 1, 3, 5):
        in_window = pairs["day_difference"].abs() <= window
        diagnostics[f"samples_within_{window}_days"] = int(
            pairs.loc[in_window, "sample_id"].nunique()
        )
    return pairs, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--cloud-threshold", type=float, default=DEFAULT_CLOUD_THRESHOLD)
    parser.add_argument("--pattern", default=PIXEL_FILE_PATTERN,
                        help="File pattern of the exports, e.g. 's2_l2a_pixel_matchups_*.csv'")
    args = parser.parse_args(argv)

    pixels = load_pixels(args.input_dir, args.pattern)
    matchups, diagnostics = build_matchups(
        pixels, min_pixels=args.min_pixels, cloud_threshold=args.cloud_threshold
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(args.output, index=False)

    for label, count in diagnostics.items():
        print(f"{label}: {count:,}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
