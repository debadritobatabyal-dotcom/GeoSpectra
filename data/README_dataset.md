# Sausar Manganese Prospectivity Dataset — README

## What's actually in `sausar_manganese_prospectivity_dataset.csv`

**351 rows total:**
- **11 documented, confirmed manganese-associated localities** (label_status =
  `confirmed_manganese_associated`) — all 11 MOIL-operated mines in the
  Balaghat (MP) / Nagpur–Bhandara (Maharashtra) sector of the Sausar belt:
  Balaghat, Bharveli, Ukwa, Tirodi, Kandri, Mansar (Munsar), Beldongri,
  Gumgaon, Chikla, Sitapatore, Dongri Buzurg. Each row has real coordinates
  (or the best available approximation, clearly flagged in
  `coordinate_precision`), real geological attributes (formation, host rock,
  ore type, age, metamorphic grade, structural notes) taken from cited
  sources, and a `data_source`/`source_url` column.
- **340 background/unlabeled grid points** (label_status =
  `background_unlabeled`) spaced ~11 km apart across a rectangular bounding
  box covering the belt corridor (20.6–22.3°N, 79.0–80.9°E). These have real
  coordinates and a real derived feature (`distance_to_nearest_known_mn_km`,
  computed by haversine distance to the 11 documented points), but **no
  geology, terrain, or satellite attributes** — see below for why.

No row was invented, and no numeric value was fabricated to "fill" a column.
Where I could not verify a real number, the cell is empty.

## What I could NOT get this session, and why

I do not have network access from this sandboxed environment to Google Earth
Engine, Copernicus Data Space, USGS EarthExplorer, GSI Bhukosh/NGDR, or any
raster/vector geospatial download endpoint (the sandbox's outbound network is
restricted to package registries like PyPI/npm/GitHub, not geospatial data
services). Web search/fetch only returns page text/snippets, not usable
raster pixel values. Concretely, this means the following columns are
present in the CSV (so the schema matches what your model will need) but are
**empty for every row**, and should not be treated as "missing at random":

- `s2_B2 … s2_B12`, `ndvi`, `ndmi`, `bsi`, `iron_oxide_index` (Sentinel-2)
- `s1_VV`, `s1_VH` (Sentinel-1 SAR)
- `elevation_m`, `slope_deg`, `aspect_deg`, `curvature`, `drainage_density` (DEM-derived)
- `fault_lineament_distance_km`, `lineament_density_per_sqkm`,
  `structural_intersection_flag` (need GSI/Bhukosh fault-lineament vector layers)
- `district` for background points (no offline reverse-geocoder available)
- Precise mine-lease coordinates for Beldongri, Gumgaon, Chikla, Sitapatore
  (only district-level location documented in the literature I could access
  this session — do not snap these to a district centroid or invent a point)
- Kajlidongri and other Jhabua-belt manganese localities were deliberately
  **excluded** — they belong to the Aravalli Supergroup's Jhabua Mn belt, a
  different geological setting from the Sausar belt you specified.

## How to actually populate the satellite/terrain columns (you have access I don't)

Since you'll be building this for the hackathon, you almost certainly do
have Earth Engine / QGIS access. Recommended fill order:

1. **Elevation/slope/aspect/curvature** — SRTM 30m (`USGS/SRTMGL1_003` in
   GEE) or Cartosat DEM from Bhoonidhi. Compute slope/aspect/curvature with
   `ee.Terrain.slope/aspect` or QGIS raster terrain tools.
2. **Sentinel-2** — pull a cloud-filtered median composite
   (`COPERNICUS/S2_SR_HARMONIZED`, <10% cloud, dry-season months) per point,
   sample bands B2–B12 at each lat/lon, then derive NDVI, NDMI, BSI, and an
   iron-oxide index (B4/B2) yourself — this is standard and reproducible.
3. **Sentinel-1** — `COPERNICUS/S1_GRD`, VV/VH, again as a temporal median.
4. **Faults/lineaments** — digitize or download the GSI Bhukosh 1:50,000
   geological/structural map vector layer for Balaghat/Nagpur/Bhandara/
   Chhindwara sheets (requires Bhukosh/NGDR registration), then compute
   point-to-line distance and a kernel-density lineament map in QGIS.
5. Re-run distance/derived-feature calculations after adding real fault
   layers — the `distance_to_nearest_known_mn_km` column already shows the
   pattern to follow.

## Expanding the positive-sample list further

11 confirmed localities is realistic for what I could source and verify
this session, but it is a small positive set for ML. To responsibly grow it
without fabricating anything:
- GSI/NGDR mineral-occurrence layers (via Bhukosh/NGDR portal — needs
  account registration, not just automated fetch) list many more Mn
  occurrences/prospects beyond the 11 active MOIL mines.
- IBM's National Mineral Inventory and state Directorate of Geology & Mining
  (Maharashtra/MP) mining-lease registers list smaller/inactive Mn leases.
- Environmental clearance / mining-plan PDFs on `environmentclearance.nic.in`
  and `nmet.gov.in` (like the Bharveli one used here) often contain exact
  boundary-pillar coordinates for individual leases — worth systematically
  scraping for every Balaghat/Bhandara/Nagpur Mn lease.

Treat any location without a documented occurrence as **unlabeled**, never
as a confirmed negative — this matters for whatever PU-learning or
one-class approach you use downstream.
