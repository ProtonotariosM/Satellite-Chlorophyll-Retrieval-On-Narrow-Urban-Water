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

# Sentinel-2 collections: Level-1C (top of atmosphere, as in the paper) and
# Level-2A (atmospherically corrected by ESA's Sen2Cor; available from 2017).
S2_COLLECTIONS = {
    "L1C": "COPERNICUS/S2_HARMONIZED",
    "L2A": "COPERNICUS/S2_SR_HARMONIZED",
}
L2A_START_DATE = "2017-04-01"

# Cloud Score+ cloud mask. QA60 is masked between Feb 2022 and Feb 2024 in
# Earth Engine, so it cannot be used for the whole 2016-2025 period.
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"
CLEAR_THRESHOLD = 0.60   # pixels with a clear-sky score >= 0.60 are kept
BUFFER_METERS = 15
# Extract the widest window once (+-5 days); narrower windows (0, 1, 3)
# are filtered afterwards using day_difference (RQ3).
MATCH_WINDOW_DAYS = 5
SPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B8", "B8A"]

def initialize_earth_engine(project: str | None = None) -> None:
    """Initialize the Earth Engine Python client."""
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()



def add_cloud_mask(image: ee.Image) -> ee.Image:
    """Mask cloudy pixels with Cloud Score+.

    The original paper used QA60, but QA60 is masked between Feb 2022 and
    Feb 2024 in Earth Engine, which would silently remove all images from
    those years. Cloud Score+ covers the whole period and also flags
    cloud shadows.
    """
    clear = image.select(CLOUD_SCORE_BAND).gte(CLEAR_THRESHOLD)
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
    """Apply cloud masking, SIAC and add indices."""
    image = add_cloud_mask(image)  #add_qa60_mask(image)
    # TODO: integrate SIAC before calling add_indices
    image = add_indices(image)
    return image



def get_candidate_images(sample_date: ee.Date, region: ee.Geometry,
                         window_days: int = MATCH_WINDOW_DAYS,
                         product: str = "L1C") -> ee.ImageCollection:
    """Get images near the sample date that intersect its station buffer"""
    start = sample_date.advance(-window_days, "day")
    # Earth Engine filterDate uses an exclusive end date.
    end = sample_date.advance(window_days + 1, "day")

    return (
        ee.ImageCollection(S2_COLLECTIONS[product])
        .filterDate(start, end)
        .filterBounds(region)
        # Attach the Cloud Score+ band to each matching Sentinel-2 image
        .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), [CLOUD_SCORE_BAND])
        .map(prepare_image)
    )


def make_pixel_features(sample: ee.Feature, window_days: int = MATCH_WINDOW_DAYS,
                        scale_m: int = 10, product: str = "L1C") -> ee.FeatureCollection:
    """Return pixel-level features for all candidate images for one sample.

    Runs on the Earth Engine server: `sample` is an ee.Feature whose geometry
    is the station point and whose properties are the in-situ values.
    """
    region = sample.geometry().buffer(BUFFER_METERS)
    sample_date = ee.Date(sample.get("date"))

    images = get_candidate_images(sample_date, region, window_days=window_days, product=product)

    def extract_image(image: ee.Image) -> ee.FeatureCollection:
        image_time = ee.Date(image.get("system:time_start"))
        # Compare calendar dates only (the image time is ~15:45 UTC, the
        # sample date is midnight), so a same-day match gives exactly 0.
        image_date = ee.Date(image_time.format("YYYY-MM-dd"))

        # We can scale Level-1C reflectance bands to reflectance-like values for use in models.
        # Indices are ratios, so their values are unchanged by this scaling.
        reflectance = image.select(SPECTRAL_BANDS).multiply(0.0001)
        index_bands = image.select(["NDVI", "NDCI", "B8AB4", "B3B2"])
        # output_image = reflectance.addBands(index_bands)
        # Keep the cloud score per pixel, so a stricter threshold can be applied later
        output_image = reflectance.addBands(index_bands).addBands(image.select(CLOUD_SCORE_BAND))
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
            scale=10,  
            maxPixels=100,
        )

        def add_metadata(feature: ee.Feature) -> ee.Feature:
            coords = feature.geometry().coordinates()
            return feature.set({
                
                "sample_id": sample.get("sample_id"),
                "station": sample.get("station"),
                "sample_date": sample_date.format("YYYY-MM-dd"),
                "image_date": image_date.format("YYYY-MM-dd"),
                "image_time_utc": image_time.format("YYYY-MM-dd HH:mm"),
                "day_difference": image_date.difference(sample_date, "day").round(),
                "chl": sample.get("chl"),
                "station_lat": sample.get("lat"),
                "station_lon": sample.get("lon"),
                "pixel_lon": coords.get(0),
                "pixel_lat": coords.get(1),
                "buffer_m": BUFFER_METERS,
                "processing": f"{product}_CloudScorePlus",  
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


def samples_to_feature_collection(samples: pd.DataFrame) -> ee.FeatureCollection:
    """Convert the sample table into an Earth Engine FeatureCollection, so the
    extraction runs on the server instead of building one request per sample."""
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row.lon), float(row.lat)]),
            {
                "sample_id": row.sample_id,
                "station": row.station,
                "date": row.date.strftime("%Y-%m-%d"),
                "chl": float(row.chl),
                "lat": float(row.lat),
                "lon": float(row.lon),
            },
        )
        for row in samples.itertuples()
    ]
    return ee.FeatureCollection(features)


def extract_pixels(samples: pd.DataFrame, window_days: int,
                   scale_m: int, product: str = "L1C") -> ee.FeatureCollection:
    """Pixel-level matchups for all samples, computed on the server."""
    sample_fc = samples_to_feature_collection(samples)
    return sample_fc.map(
        lambda sample: make_pixel_features(sample, window_days=window_days,
                                           scale_m=scale_m, product=product)
    ).flatten()


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


    parser.add_argument(
        "--limit", type=int, default=None,
        help="Test mode: process only the first N samples and save the result locally",
    )
    parser.add_argument(
        "--stations", nargs="*", default=None,
        help="Only use these stations (e.g. --stations NC1 E2)",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Only export this year (default: one export task per year)",
    )

    parser.add_argument(
        "--product", choices=sorted(S2_COLLECTIONS), default="L1C",
        help="L1C = top of atmosphere (paper); L2A = atmospherically corrected (from 2017)",
    )

    args = parser.parse_args()

    initialize_earth_engine(args.project)
    samples = load_samples(args.input)
    if args.stations:
        samples = samples[samples["station"].isin(args.stations)]
    if args.year:
        samples = samples[samples["date"].dt.year == args.year]
    if args.product == "L2A":
        # Level-2A images are only available in Earth Engine from 2017
        samples = samples[samples["date"] >= L2A_START_DATE]
    if samples.empty:
        raise RuntimeError("No valid insitu samples to process.")

    # Test mode: compute a few samples directly and save them locally.
    if args.limit:
        test = samples.head(args.limit)
        result = extract_pixels(test, args.window_days, args.scale_m, args.product).getInfo()
        rows = pd.DataFrame([f["properties"] for f in result["features"]])
        out = Path("data/processed") / f"{args.output_prefix}_test.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        print(f"{len(test)} samples -> {len(rows)} pixel rows. Saved: {out}")
        return

    # Full run: one Drive export task per year keeps each request small.
    for year, year_samples in samples.groupby(samples["date"].dt.year):
        name = f"{args.output_prefix}_{year}"
        task = ee.batch.Export.table.toDrive(
            collection=extract_pixels(year_samples, args.window_days, args.scale_m, args.product),
            description=name,
            fileNamePrefix=name,
            fileFormat="CSV",
        )
        task.start()
        print(f"Started Earth Engine export task {name}: {task.id} ({len(year_samples)} samples)")


if __name__ == "__main__":
    main()