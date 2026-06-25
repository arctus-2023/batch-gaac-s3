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
   - *Snow/ice*: dual Otsu threshold on NDSI and Oa02 (blue), computed from permanent water pixels in the classification band; falls back to fixed thresholds if the classification band is absent
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

### Products

| Tier | Description | Files |
|------|-------------|-------|
| **PMP** | Per-scene, per-variable GeoTIFF | 1 band, float32, nodata=NaN |
| **DRP daily** | Temporal merge of same-day PMPs (mean / std / count) | 3 bands |
| **DRP monthly** | Count-weighted pool of daily DRPs | 3 bands |
| **DRP yearly** | Count-weighted pool of monthly DRPs | 3 bands |

Outliers (values outside the [5 %, 95 %] percentile of finite pixels in each input file) are excluded before accumulation.

### Algorithms

| Variable | Units | Algorithm key | Method |
|----------|-------|---------------|--------|
| Chla | mg m⁻³ | `gons2005` | NIR-red (665/709), CDOM-insensitive |
| | | `oc4me` | OC4Me log-polynomial (Rrs 443/490/510/560) |
| | | `ndci` | NDCI (665/709) |
| CDOM | m⁻¹ | `mabit2022` | Power-law band ratio (443/560) |
| | | `glukhovets2020` | Log-linear band ratio (443/490) |
| SPM | g m⁻³ | `dogliotti2015` | Red/NIR switching (665/865 nm) |
| | | `nechad2010` | Single-band (665 nm, switches to 865 nm) |
| | | `doxaran2012` | NIR/green ratio (865/560 nm) |
| Turbidity | FNU | `dogliotti2015_t` | Same switching scheme as dogliotti2015 SPM |
| | | `nechad2016_olci` | Multi-band OLCI LUT (665/709/865 nm) |

### Usage

```bash
python batch_wq.py wq_config.yml [options]
```

#### Options

| Flag | Description |
|------|-------------|
| `--scene SUBSTR` | Process only scenes whose directory name contains `SUBSTR` (repeatable) |
| `--pmp-only` | Compute scene-level PMP products only; skip DRP aggregation |
| `--drp-only` | Run DRP aggregation only (PMPs must already exist) |
| `--period daily\|monthly\|yearly` | Restrict DRP aggregation to one period tier |
| `--limit N` | Stop after processing N scenes |

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
```

### Configuration

Copy and edit `wq_config.yml`:

```yaml
gaac_gen_dir: /path/to/gaac_gen/src  # optional; enables Cinputmask for mask reading

l2_dir: /path/to/L2_output           # directory containing *_GAAC/ scene subdirs
l3_dir: /path/to/L3_output           # root for PMP + DRP product tree
aoi_name: JamesBay                   # label used in DRP filenames

wq_products:
  chla:
    enabled: true
    algorithm: gons2005
    params: {}                        # optional coefficient overrides
  cdom:
    enabled: true
    algorithm: mabit2022
    params: {}
  spm:
    enabled: true
    algorithm: dogliotti2015
    params: {}
  turbidity:
    enabled: true
    algorithm: dogliotti2015_t
    params: {}

aggregation:
  periods: [daily, monthly, yearly]

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
