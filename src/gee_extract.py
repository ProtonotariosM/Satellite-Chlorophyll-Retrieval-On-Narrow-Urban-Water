"""Sentinel-2 index extraction in Google Earth Engine
Extract pixel-level Sentinel-2 observations around in-situ samples.

TODO: Integrate SIAC before calling add_indices() for the original 
paper's corrected reflectance.
Python implementation can be found here:
https://github.com/MarcYin/SIAC

Expected input CSV columns:
    station, date, lat, lon, chl

Output:
    One feature per valid image pixel within the station buffer, with the
    in-situ sample and image-match metadata repeated on each pixel row.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import ee
import pandas as pd

# Sentinel-2 Level-1C collection. 
S2_COLLECTION = "COPERNICUS/S2_HARMONIZED"
BUFFER_METERS = 15   
MATCH_WINDOW_DAYS = 3
SPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B8", "B8A"]

def initialize_earth_engine(project: str | None = None) -> None:
    """Initialize the Earth Engine Python client."""
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def add_qa60_mask(image: ee.Image) -> ee.Image:
    """Mask pixels flagged as opaque cloud or cirrus by QA60. Reproduction
    from the paper."""
    qa = image.select("QA60")
    opaque_cloud = 1 << 10
    cirrus = 1 << 11

    clear = (
        qa.bitwiseAnd(opaque_cloud).eq(0)
        .And(qa.bitwiseAnd(cirrus).eq(0))
    )
    return image.updateMask(clear)


def add_indices(image: ee.Image) -> ee.Image:
    """Formulas are specified in Table 1 of the original paper.

    NDVI = (B8 - B4) / (B8 + B4)
    NDCI = (B5 - B4) / (B5 + B4)
    B8AB4 = (B8A - B4) / (B8A + B4)
    B3B2 = (B3 - B2) / (B3 + B2)
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndci = image.normalizedDifference(["B5", "B4"]).rename("NDCI")
    b8ab4 = image.normalizedDifference(["B8A", "B4"]).rename("B8AB4")
    b3b2 = image.normalizedDifference(["B3", "B2"]).rename("B3B2")
    
    return image.addBands([ndci, ndvi, b8ab4, b3b2])


def prepare_image(image: ee.Image) -> ee.Image:
    """Apply QA60 masking, SIAC and add indices."""
    image = add_qa60_mask(image)
    # TODO: integrate SIAC before calling add_indices
    image = add_indices(image)
    return image

def get_candidate_images(sample_date: pd.Timestamp, point: ee.Geometry.Point,
                         window_days: int = MATCH_WINDOW_DAYS) -> ee.ImageCollection:
    """Get images near the sample date that intersect its station buffer"""
    sample_date = pd.Timestamp(sample_date)
    start = (sample_date - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    # Earth Engine filterDate uses an exclusive end date.
    end = (sample_date + pd.Timedelta(days=window_days + 1)).strftime("%Y-%m-%d")

    region = point.buffer(BUFFER_METERS)
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start, end)
        .filterBounds(region)
        .map(prepare_image)
    )

def make_pixel_features(sample: dict, window_days: int = MATCH_WINDOW_DAYS,
                        scale_m: int = 10) -> ee.FeatureCollection:
    """Return pixel-level features for all candidate images for one sample."""
    point = ee.Geometry.Point([float(sample['lon']), float(sample['lat'])])
    region = point.buffer(BUFFER_METERS)

    images = get_candidate_images(
        pd.Timestamp(sample["date"]),
        point,
        window_days=window_days
    )

    def extract_image(image: ee.Image) -> ee.FeatureCollection:
        image_date = ee.Date(image.get("system:time_start"))
        sample_date = ee.Date(pd.Timestamp(sample["date"]).strftime("%Y-%m-%d"))

        # We can scale Level-1C reflectance bands to reflectance-like values for use in models.
        # Indices are ratios, so their values are unchanged by this scaling.
        reflectance = image.select(SPECTRAL_BANDS).multiply(0.0001)
        index_bands = image.select(["NDVI", "NDCI", "B8AB4", "B3B2"])
        output_image = reflectance.addBands(index_bands)
        pixels = output_image.sample(
            region=region,
            scale=scale_m,
            geometries=True,
            tileScale=4,  # save computation
        )

        buffer_means = output_image.select(
            ["NDVI", "NDCI", "B8AB4", "B3B2"]
        ).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            # B5 AND B8A are 20 m resolution, the rest are 10 m. The reducer’s scale affects the mean.
            scale=10,  # TODO: choose and document the baseline scale
            maxPixels=100,
        )

        def add_metadata(feature: ee.Feature) -> ee.Feature:
            coords = feature.geometry().coordinates()
            return feature.set({
                "sample_id": sample['sample_id'],
                "station": sample["station"],
                "sample_date": sample_date.format("YYYY-MM-dd"),
                "image_date": image_date.format("YYYY-MM-dd"),
                "day_difference": image_date.difference(sample_date, "day"),
                "chl": float(sample["chl"]),
                "station_lat": float(sample["lat"]),
                "station_lon": float(sample["lon"]),
                "pixel_lon": coords.get(0),
                "pixel_lat": coords.get(1),
                "buffer_m": BUFFER_METERS,
                "processing": "L1C_QA60_SIAC",
                "image_id": image.get("system:index"),
                "NDVI_buffer_mean": buffer_means.get("NDVI"),   # keep the pixel-level data, but also calculate mean for the baseline
                "NDCI_buffer_mean": buffer_means.get("NDCI"),
                "B8AB4_buffer_mean": buffer_means.get("B8AB4"),
                "B3B2_buffer_mean": buffer_means.get("B3B2"),
            })

        return pixels.map(add_metadata)

    return ee.FeatureCollection(images.map(extract_image).flatten())

def load_samples(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned station-date table and validate."""
    samples = pd.read_csv(csv_path, parse_dates=["date"])

    required = {"station", "date", "lat", "lon", "chl"}
    missing  = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    samples = samples.dropna(subset=["station", "date", "lat", "lon", "chl"]).copy()
    samples = samples.reset_index(drop=True)
    samples["sample_id"] = samples.apply(
        lambda row: f"{row["station"]}_{row["date"]:%Y%m%d}_{row.name}",
        axis=1
    )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/insitu_station_date.csv"))
    parser.add_argument("--project", default="alien-vim-299320", help="Google Cloud project for Earth Engine")
    parser.add_argument("--window-days", type=int, default=MATCH_WINDOW_DAYS)
    parser.add_argument("--scale-m", type=int, default=10)
    parser.add_argument(
        "--output-prefix",
        default="s2_pixel_matchups",
        help="Prefix for the Earth Engine Drive export task",
    )
    args = parser.parse_args()

    initialize_earth_engine(args.project)
    samples = load_samples(args.input)

    collections = []
    for sample in samples.to_dict(orient="records"):
        collections.append(
            make_pixel_features(sample, window_days=args.window_days, scale_m=args.scale_m)
        )

    if not collections:
        raise RuntimeError("No valid insitu samples to process.")

    all_pixels = ee.FeatureCollection(collections).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=all_pixels,
        description=args.output_prefix,
        fileNamePrefix=args.output_prefix,
        fileFormat="CSV",
    )
    task.start()
    print(f"Started Earth Engine export task: {task.id}")


if __name__ == "__main__":
    main()