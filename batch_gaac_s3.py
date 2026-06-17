"""
Batch GAAC atmospheric correction for Sentinel-3 OLCI L1 GeoTIFF images.

Usage:
    python batch_gaac_s3.py config.yml
    python batch_gaac_s3.py config.yml --dry-run   # list scenes, no processing
"""

from __future__ import annotations

import argparse
import copy
import glob
import logging
import os
import sys

import numpy as np
import rasterio
import yaml


# ── Classification band helpers ───────────────────────────────────────────────
# Classification encoding: 0=clear_land, 1=clear_water,
#                          2=cloud_land,  3=cloud_water, 255=invalid

def get_clear_water_pct(tif_path: str) -> float | None:
    """Return the fraction (0–100) of clear-water pixels in the classification band.

    Returns None if the file has no classification band.
    """
    with rasterio.open(tif_path) as src:
        last_desc = src.descriptions[-1] or ''
        if 'classification' not in str(last_desc).lower():
            return None
        class_data = src.read(src.count)

    n_clear_water = (class_data == 1).sum()
    n_cloud_water = (class_data == 3).sum()
    n_total_water = n_clear_water + n_cloud_water
    if n_total_water == 0:
        return 0.0
    return 100.0 * n_clear_water / n_total_water


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_parameters(cfg: dict) -> dict:
    """Translate YAML config into the kwargs dict expected by gaac.main.run()."""
    from skimage.morphology import square

    masking_cfg = cfg['masking'].copy()
    selem_size  = masking_cfg.pop('selem_size', 5)
    masking_cfg['selem'] = square(selem_size)

    aerosol_cfg = cfg['aerosol'].copy()
    raw_best    = aerosol_cfg.get('best_ind', [])
    aerosol_cfg['best_ind'] = tuple(raw_best) if raw_best else ()

    return {
        'masking':  masking_cfg,
        'rayleigh': cfg['rayleigh'].copy(),
        'aerosol':  aerosol_cfg,
    }


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'batch_gaac_s3.log')

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('batch_gaac_s3')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Main batch loop ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Batch GAAC for Sentinel-3 OLCI L1 images')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--dry-run', action='store_true',
                        help='List scenes and clear-water pct without processing')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after processing this many scenes (useful for testing)')
    parser.add_argument('--scene', action='append', default=None, metavar='SUBSTR',
                        help='Only process scenes whose filename contains SUBSTR '
                             '(repeatable; any match includes the scene)')
    parser.add_argument('--ndwi-threshold', type=float, default=None, metavar='T',
                        help='Override the NDWI water-mask threshold from the config '
                             '(e.g. 0.3); pixels with NDWI > T are classified as water')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Inject gaac_gen into sys.path before any gaac imports ─────────────────
    gaac_gen_dir = cfg['gaac_gen_dir']
    if not os.path.isdir(os.path.join(gaac_gen_dir, 'gaac')):
        raise FileNotFoundError(
            f"gaac_gen_dir '{gaac_gen_dir}' does not contain a 'gaac' package. "
            f"Set gaac_gen_dir to the 'src' subdirectory of the gaac_gen repo "
            f"(e.g. /path/to/gaac_gen/src)."
        )
    if gaac_gen_dir not in sys.path:
        sys.path.insert(0, gaac_gen_dir)

    acolite_dir = cfg.get('acolite_dir', '')
    if acolite_dir:
        os.environ.setdefault('gaac_acolite_dir', acolite_dir)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    l1_dir    = cfg['l1_dir']
    out_dir   = cfg['output_dir']
    threshold = float(cfg.get('clear_water_threshold', 5.0))
    input_type = cfg.get('input_type', 'ACOTOA')

    if args.ndwi_threshold is not None:
        cfg['masking']['threshold'] = args.ndwi_threshold

    logger = setup_logging(out_dir)
    logger.info(f'Config       : {args.config}')
    logger.info(f'L1 directory : {l1_dir}')
    logger.info(f'Output dir   : {out_dir}')
    logger.info(f'Input type   : {input_type}')
    logger.info(f'Clear-water threshold: {threshold:.1f} %')
    logger.info(f'NDWI threshold: {cfg["masking"]["threshold"]}')

    scenes = sorted(glob.glob(os.path.join(l1_dir, '*.tif')))
    if not scenes:
        logger.warning(f'No .tif files found in {l1_dir}')
        return

    logger.info(f'Found {len(scenes)} .tif scene(s)')

    cgaac_kwargs = cfg.get('cgaac', {})

    # Build parameters lazily on first processing call (defers skimage import).
    _parameters = None

    skipped_threshold = []
    skipped_error     = []
    processed         = []

    for scene_f in scenes:
        scene_name = os.path.basename(scene_f)

        # ── Scene name filter ──────────────────────────────────────────────────
        if args.scene and not any(s in scene_name for s in args.scene):
            continue

        # ── Clear-water filter ─────────────────────────────────────────────────
        cw_pct = get_clear_water_pct(scene_f)
        if cw_pct is None:
            logger.warning(f'[SKIP] {scene_name}: no classification band found')
            skipped_threshold.append(scene_name)
            continue

        logger.info(f'Scene: {scene_name}  clear-water={cw_pct:.1f}%')

        if cw_pct < threshold:
            logger.info(f'  → below threshold ({threshold:.1f}%), skipping')
            skipped_threshold.append(scene_name)
            continue

        if args.dry_run:
            logger.info(f'  → [DRY-RUN] would process')
            processed.append(scene_name)
            continue

        # ── Run GAAC ──────────────────────────────────────────────────────────
        # Defer heavy imports until first actual processing scene so that
        # --dry-run and the clear-water filter work without TensorFlow/skimage.
        if _parameters is None:
            _parameters = build_parameters(cfg)
            from gaac.ac.main import Cgaac
            from gaac.main import run

        logger.info(f'  → processing …')
        try:
            gaac = Cgaac(**cgaac_kwargs)
            run(input_f=scene_f,
                gaac=gaac,
                opt_pixles=None,
                input_type=input_type,
                output_dir=out_dir,
                **copy.deepcopy(_parameters))
            processed.append(scene_name)
            logger.info(f'  → done')
        except Exception as exc:
            logger.error(f'  → FAILED: {exc}', exc_info=True)
            skipped_error.append(scene_name)

        if args.limit and len(processed) >= args.limit:
            logger.info(f'Reached --limit {args.limit}, stopping.')
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info('')
    logger.info('===== Batch summary =====')
    logger.info(f'  Processed : {len(processed)}')
    logger.info(f'  Skipped (threshold) : {len(skipped_threshold)}')
    logger.info(f'  Failed    : {len(skipped_error)}')
    if skipped_error:
        for s in skipped_error:
            logger.info(f'    FAILED: {s}')


if __name__ == '__main__':
    main()
