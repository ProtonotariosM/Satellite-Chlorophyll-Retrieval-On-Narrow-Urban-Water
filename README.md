# Narrow Water, Mixed Pixels

Urban Computing 2026–2027 project, group 14: Dominika Haik, Michail Protonotarios.

**Research question:** does Sentinel-2 chlorophyll-a retrieval still work in narrow urban waterways? We compare open-water and tributary stations in New York Harbor and test fixes for mixed water/land pixels.

Based on: Kislik et al. (2022), *Mapping algal bloom dynamics in small reservoirs using Sentinel-2 imagery in Google Earth Engine*, Ecological Indicators 140, 109041. https://doi.org/10.1016/j.ecolind.2022.109041. Original code: https://github.com/chippiekizzle/Klamath_Reservoirs

## Status
Proposal submitted (3 Nov 2026). No code yet.

## Planned structure
```
data/          raw and processed data (not committed, see below)
notebooks/     exploration and figures
src/
  insitu.py      clean NYC harbor data, station table
  watermask.py   water fractions and channel widths from OpenStreetMap
  gee_extract.py Sentinel-2 index extraction in Google Earth Engine
  mixing.py      purity filter and linear unmixing
  models.py      regression and cross-validation
results/       figures and tables for the report
```

## Data
- In-situ chlorophyll-a: NYC DEP, *Harbor Water Quality*, NYC Open Data, https://data.cityofnewyork.us/d/5uug-f49n. A snapshot (downloaded November 2026) is included in data/raw/harbor_water_quality.csv.
- Sentinel-2: accessed through Google Earth Engine (no download needed).
- Water polygons: OpenStreetMap, https://download.geofabrik.de/north-america/us/new-york.html
