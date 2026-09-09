# batch-gaac-s3

Two independent pipelines for Sentinel-3 OLCI:

1. **`batch_gaac_s3.py`** — atmospheric correction (L1 → L2 water-leaving reflectance)
2. **`batch_wq.py`** — water quality retrieval (L2 → PMP/DRP products for Chla, CDOM, SPM, Turbidity)

---

## Atmospheric Correction (`batch_gaac_s3.py`)

### Overview

The pipeline processes each scene through three stages:

1. **Rayleigh + gas correction** — ACOLITE LUT-based, writes `*_rhor.tif`
2. **Masking** — combines three independent masks into a single final product (`*_mask.tif`):
   - *Water*: pure NDWI (`*_watermask.tif`, `0=null, 5=water`)
   - *Cloud*: classification band values `{2, 3}` (cloud over land or water)
   - *Snow/ice*: ENDSIII index `(Oa12−Oa16+Oa20−Oa21)/(Oa12+Oa16+Oa20+Oa21)` > threshold (fixed −0.01 by default; set `use_otsu: true` to compute via Otsu on permanent water pixels from the classification band); morphological opening/closing and small-component removal applied
3. **Aerosol correction** — GA optimisation + adjacency-effect and sky/sun-glint correction; writes `*_rhor_rhow.tif` and `*_rhor_rhoadj.tif`

When `tile_size` is set, the aerosol step runs one GA optimisation per tile and interpolates the result spatially across the scene (tiled AC). Optimization pixel locations are exported automatically as `*_rhor_opt_pixels.gpkg`.

Scenes are filtered by **clear-water percentage** before processing:

```
clear_water_pct = clear_water_pixels / (clear_water_pixels + cloud_water_pixels) × 100
```

where pixel classes come from the classification band of the L1 TOA GeoTIFF (0=clear_land, 1=clear_water, 2=cloud_land, 3=cloud_water, 255=invalid).

### Setup on a new machine

**1. Create the conda environment** (provides GDAL native libs):

```bash
conda env create -f gaac_gen/environment.yml
conda activate gaac
```

**2. Clone the batch repo and install Python dependencies:**

```bash
git clone https://github.com/arctus-2023/batch-gaac-s3.git
cd batch-gaac-s3
uv sync
```

`uv sync` uses `uv.lock` to reproduce the exact package versions. The `.venv` directory is not committed to git.

**3. Install GDAL into the venv** (links against native libs from the conda env):

```bash
PATH=$CONDA_PREFIX/bin:$PATH \
GDAL_CONFIG=$CONDA_PREFIX/bin/gdal-config \
uv pip install --python .venv "gdal==3.10.3"
```

GDAL is only required for GeoPackage opt-pixel export — the AC pipeline runs without it.

### Usage

```bash
python batch_gaac_s3.py <config.yml> [options]
```

#### Options

| Flag | Description |
|------|-------------|
| `--dry-run` | List scenes and clear-water % without processing |
| `--limit N` | Stop after processing N scenes (useful for testing) |
| `--scene SUBSTR` | Only process scenes whose filename contains `SUBSTR` (repeatable) |
| `--ndwi-threshold T` | Override the NDWI water-mask threshold from the config |

#### Examples

```bash
# Full batch run
python batch_gaac_s3.py batch_gaac_s3_config_test.yml

# Survey scenes before processing — writes dryrun_YYYYMMDD_HHMMSS.csv to output_dir
python batch_gaac_s3.py batch_gaac_s3_config_test.yml --dry-run

# Test on one scene
python batch_gaac_s3.py batch_gaac_s3_config_test.yml --limit 1

# Reprocess a specific scene with a stricter water mask
python batch_gaac_s3.py batch_gaac_s3_config_test.yml \
    --scene S3A_L1TOA_20250712 --ndwi-threshold 0.3
```

#### Dry-run CSV

`--dry-run` writes a `dryrun_YYYYMMDD_HHMMSS.csv` file to `output_dir` with one row per scene:

| Column | Description |
|--------|-------------|
| `scene` | Scene filename |
| `clear_water_pct` | `clear_water / (clear_water + cloud_water) × 100` |
| `status` | `would_process`, `below_threshold`, or `no_classification_band` |

### Configuration

Copy and edit `batch_gaac_s3_config_test.yml`:

```yaml
gaac_gen_dir: /path/to/gaac_gen/src   # must point to the src/ subdirectory
acolite_dir:  /path/to/acolite

l1_dir:    /path/to/L1/scenes
output_dir: /path/to/L2_output

input_type: ACOTOA
clear_water_threshold: 5.0   # minimum clear-water % to process a scene

masking:
  method: ndwi
  threshold: 0.0             # NDWI threshold (override with --ndwi-threshold)
  replace_output: false

rayleigh:
  proc: acolite
  replace_output: false
  output_rgb: true
  use_ancillary: false

aerosol:
  replace_output: false
  perform_ac: true
  tile_size: 200             # remove or comment out to disable tiled AC
```

### Outputs

Each processed scene produces a `<scene_name>_GAAC/` subdirectory containing:

| File | Description |
|------|-------------|
| `*_rhor.tif` | Rayleigh-corrected reflectance |
| `*_watermask.tif` | Pure NDWI water mask (`0=null, 5=water`) |
| `*_mask.tif` | Final combined mask (NDWI + cloud + snow); S3 scheme: `0=clear water, 1=cloud/water, 2=snow/water, 3=cloud+snow/water`, `50–53` over permanent land, `255=nodata` |
| `*_rhor_rhow.tif` | Water-leaving reflectance |
| `*_rhor_rhoadj.tif` | Adjacency-corrected reflectance |
| `*_rhor_rgb.tif` | RGB preview with optimization pixel(s) marked |
| `*_rhor_opt_pixels.gpkg` | Optimization pixel locations (GeoPackage) |
| `*_tile_NN_res.png` | Per-tile GA optimization fit plots (tiled AC only) |
| `log_gaac_*.txt` | Processing log |

---

## Water Quality Retrieval (`batch_wq.py`)

Takes `*_GAAC/` directories produced by `batch_gaac_s3.py` and retrieves four water quality variables from the water-leaving reflectance (`*_rhor_rhow.tif`).

### Supported sensors

| Family | Platforms | Water bands in the L2 file |
|--------|-----------|----------------------------|
| `OLCI` | Sentinel-3A / 3B | 400 412 443 490 510 560 620 665 674 682 709 754 779 865 884 |
| `MSI` | Sentinel-2A / 2B / 2C | 442 492 559 665 704 739 780 833 864 |
| `OLI` | Landsat-8 OLI, Landsat-9 OLI-2 | 443 482 561 655 865 |

Each scene's instrument comes from the `sensor` tag the GAAC writer stamps on the rhow file, falling back to the scene name. `l2_dir` is searched **recursively**, so per-mission subdirectories (`L2/S2/`, `L2/L8/`, …) are picked up alongside scenes at the top level.

Algorithms declare *nominal* wavelengths — the OLCI-centric numbers used in the literature — and each is matched to the nearest band the scene actually carries, within `band_tolerance` (default ±15 nm). So an algorithm asking for 665 nm reads OLI's 655 nm band and one asking for 709 nm reads MSI's 704 nm band, while a request with no counterpart (510 nm on MSI/OLI) is reported missing and that product is skipped for the scene with a warning. Substitutions are logged per scene.

### Products

| Tier | Description | Files |
|------|-------------|-------|
| **PMP** | Per-scene, per-variable GeoTIFF | 1 band, float32, nodata=NaN |
| **DRP daily** | Temporal merge of same-day PMPs (mean / std / count) | 3 bands |
| **DRP monthly** | Count-weighted pool of daily DRPs | 3 bands |
| **DRP yearly** | Count-weighted pool of monthly DRPs | 3 bands |

Outliers (values outside the [5 %, 95 %] percentile of finite pixels in each input file) are excluded before accumulation. Clipping happens at the input's native resolution *before* any warp, so extreme pixels are removed rather than smeared into coarser target cells.

### Merging across sensors

DRP composites merge every contributing mission for a period into one product — `<aoi>_<date>_<var>.tif`. Since the missions do not share a grid, each PMP is warped onto a common target grid first, set by `aggregation.grid`:

```yaml
aggregation:
  sensors: all                 # or [MSI, OLI] / [LANDSAT] / [S2B] — see below
  grid:
    crs: EPSG:32617
    resolution: 30             # metres; single value or [xres, yres]
    bounds: [596055, 5932935, 649125, 5999145]   # or `auto` = union of inputs
    resampling: average        # average | bilinear | cubic | nearest | mode | min | max | med
```

Fixing the grid in the config keeps composites comparable across runs even as the set of available scenes changes — prefer it to `bounds: auto`, which lets the extent move between runs. Omit the `grid` block entirely and inputs that already share a grid are composited in place with no warping at all; a mixed set then falls back to the finest contributing resolution, with a warning.

`average` is the right resampling for downsampling continuous fields; NaN pixels are excluded from each average rather than counted as zero. An input already on the target grid is passed through untouched.

`aggregation.sensors` restricts which sensors take part. Selectors work at four granularities, and `--sensor` on the command line overrides the config for one run:

| Granularity | Examples |
|---|---|
| Exact sensor tag | `S3_OLCIA`, `S2_MSIB`, `L8_OLI`, `L9_OLI2` |
| Platform | `S3A`, `S2B`, `L8`, `L9` |
| Mission series | `S3`, `S2`, `LANDSAT` |
| Instrument family | `OLCI`, `MSI`, `OLI` |

PMPs are still produced for every scene; this only filters the composites.

Merging missions can also merge *algorithms* — a product may use `ndci` on MSI and `nir_red` on OLI. That is allowed, recorded in the output's `algorithm` and `sensors` tags, and logged as a warning so it does not pass unnoticed.

### Algorithms

Wavelengths below are nominal; see band matching above. `python batch_wq.py <config> --list-algorithms` prints this table live from the registry.

| Variable | Units | Algorithm key | Sensors | Method |
|----------|-------|---------------|---------|--------|
| Chla | mg m⁻³ | `gons2005` | OLCI, MSI | NIR-red (665/709/779), CDOM-insensitive |
| | | `oc4me` | OLCI | OC4Me log-polynomial (443/490/510/560) |
| | | `ndci` | OLCI, MSI | NDCI (665/709) |
| | | `nir_red` | OLI | Linear 865/665 ratio proxy — OLI has no red edge |
| CDOM | m⁻¹ | `mabit2022` | all | Power-law band ratio (443/560) |
| | | `glukhovets2020` | all | Log-linear band ratio (443/490) |
| | | `mabit_redgreen` | all | Red/green log-power (665/560), no blue bands needed |
| SPM | g m⁻³ | `dogliotti2015` | all | Red/NIR switching (665/865) |
| | | `nechad2010` | all | Single-band (665, switches to 865) |
| | | `doxaran2012` | all | NIR/green ratio (865/560) |
| | | `mabit_powerlaw` | all | Single-band log-power — 740 nm on MSI, 665 nm on OLI (B4) and OLCI (Oa8) |
| Turbidity | FNU | `dogliotti2015_t` | all | Blended red/NIR switch (665/865) |
| | | `dogliotti2015_hs` | all | Same coefficients, hard ρw(red) < 0.05 switch |
| | | `nechad2016_olci` | OLCI | Multi-band OLCI LUT (665/709/865) |

`mabit_redgreen`, `mabit_powerlaw`, `dogliotti2015_hs`, and `nir_red` are the forms used in the SAMBA / Eeyou-Sat `l3_MSI_OLI` L3 chain, ported here. The first three reproduce that chain bit-for-bit on MSI and OLI scenes; `mabit_powerlaw` additionally runs on OLCI, reading Oa8 (665 nm) with the OLI red-band pair. `nir_red` differs on one point: it returns NaN where the retrieval is negative — which is most of a clear-water scene — instead of clamping to 0, so that DRP temporal means are not loaded with a mass of exact zeros that are not measurements.

### Usage

```bash
python batch_wq.py wq_config.yml [options]
```

#### Options

| Flag | Description |
|------|-------------|
| `--scene SUBSTR` | Process only scenes whose directory name contains `SUBSTR` (repeatable) |
| `--sensor NAME` | Process only this sensor, platform, or family — `L8`, `L9`, `OLI`, `S2`, `MSI`, `S3A`, `OLCI`, `L8_OLI`, … (repeatable) |
| `--pmp-only` | Compute scene-level PMP products only; skip DRP aggregation |
| `--drp-only` | Run DRP aggregation only (PMPs must already exist) |
| `--period daily\|monthly\|yearly` | Restrict DRP aggregation to one period tier |
| `--limit N` | Stop after processing N scenes |
| `--list-algorithms` | Print the algorithm registry with bands and supported sensors, then exit |

#### Examples

```bash
# Full run — PMP + all DRP tiers
python batch_wq.py wq_config.yml

# Single scene
python batch_wq.py wq_config.yml --scene S3A_L1TOA_20250615

# Regenerate DRP products only (after new scenes were added)
python batch_wq.py wq_config.yml --drp-only

# Daily DRP only
python batch_wq.py wq_config.yml --drp-only --period daily

# Landsat scenes only
python batch_wq.py wq_config.yml --sensor OLI

# What can run on what
python batch_wq.py wq_config.yml --list-algorithms
```

### Configuration

Copy and edit `wq_config.yml`:

```yaml
gaac_gen_dir: /path/to/gaac_gen/src  # optional; enables Cinputmask for mask reading

l2_dir: /path/to/L2_output           # searched recursively for *_GAAC/ scene dirs
l3_dir: /path/to/L3_output           # root for PMP + DRP product tree
aoi_name: JamesBay                   # label used in DRP filenames

band_tolerance: 15                    # nm; nominal→actual band matching window

# `sensors:` picks a different algorithm per mission. Keys are matched
# most-specific-first: exact sensor tag (S3_OLCIA, S2_MSIB, L8_OLI, L9_OLI2),
# then platform (S3A, S2, L8, L9), then family (OLCI, MSI, OLI). A value may be
# a bare algorithm name or a full {algorithm, params, enabled} mapping.
wq_products:
  chla:
    enabled: true
    algorithm: gons2005               # the default, used where no sensor matches
    params: {}                        # optional coefficient overrides
    sensors:
      MSI: ndci
      OLI: nir_red
  cdom:
    enabled: true
    algorithm: mabit2022
    params: {}
    sensors:
      MSI: mabit_redgreen
      OLI: mabit_redgreen
  spm:
    enabled: true
    algorithm: dogliotti2015
    params: {}
    sensors:
      MSI: {algorithm: mabit_powerlaw, params: {MSI_A: 17.0, MSI_B: 0.42}}
      OLI: mabit_powerlaw
  turbidity:
    enabled: true
    algorithm: dogliotti2015_t
    params: {}
    sensors:
      MSI: dogliotti2015_hs
      OLI: dogliotti2015_hs

aggregation:
  periods: [daily, monthly, yearly]
  sensors: all                        # which sensors enter the composites
  grid:                               # common grid every PMP is warped onto
    crs: EPSG:32617
    resolution: 30
    bounds: [596055, 5932935, 649125, 5999145]
    resampling: average

replace_output: false                 # set true to overwrite existing outputs
```

### Output structure

```
<l3_dir>/
├── PMP/
│   └── YYYY/MM/DD/
│       └── <scene_stem>/
│           ├── <scene_stem>_chla.tif
│           ├── <scene_stem>_cdom.tif
│           ├── <scene_stem>_spm.tif
│           └── <scene_stem>_turbidity.tif
└── DRP/
    ├── daily/YYYY/MM/DD/
    │   └── <aoi>_YYYYMMDD_<var>.tif        (bands: mean / std / count)
    ├── monthly/YYYY/MM/
    │   └── <aoi>_YYYYMM_<var>.tif
    └── yearly/YYYY/
        └── <aoi>_YYYY_<var>.tif
```

Every DRP carries `sensors`, `grid`, and `n_scenes`/`n_days`/`n_months` tags recording what went into it. All DRPs sit on the configured target grid regardless of which missions contributed.
