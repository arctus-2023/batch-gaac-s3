#!/usr/bin/env python3
"""Water Quality retrieval pipeline for Sentinel-3 OLCI L2 scenes.

Reads atmospherically corrected water-leaving reflectance (*_rhor_rhow.tif)
from the GAAC batch pipeline and retrieves CDOM, Chla, SPM, and Turbidity.

Products
--------
PMP (Primary): one GeoTIFF per variable per scene
DRP daily    : temporal merge of same-day PMPs (mean / std / count)
DRP monthly  : pooled aggregate of daily DRPs
DRP yearly   : pooled aggregate of monthly DRPs

Usage
-----
# Full run (PMP + all DRP tiers):
    python batch_wq.py wq_config.yml

# Single scene (substring match on directory name):
    python batch_wq.py wq_config.yml --scene S3A_L1TOA_20250615

# PMP only (no aggregation):
    python batch_wq.py wq_config.yml --pmp-only

# DRP only (requires PMPs to already exist):
    python batch_wq.py wq_config.yml --drp-only

# Specific DRP period:
    python batch_wq.py wq_config.yml --period daily

# Limit number of scenes processed:
    python batch_wq.py wq_config.yml --limit 3
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path


def _setup_logging(log_path: str) -> None:
    fmt = '%(asctime)s %(levelname)-8s %(name)s — %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding='utf-8'),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Water quality retrieval for Sentinel-3 OLCI L2 products',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('config', help='Path to wq_config.yml')
    parser.add_argument(
        '--scene', '-s', action='append', default=None, metavar='STR',
        help='Process only scene dirs whose name contains STR (repeatable)',
    )
    parser.add_argument(
        '--pmp-only', action='store_true',
        help='Only compute scene-level PMP products; skip DRP aggregation',
    )
    parser.add_argument(
        '--drp-only', action='store_true',
        help='Only run DRP aggregation (PMPs must already exist)',
    )
    parser.add_argument(
        '--period', choices=['daily', 'monthly', 'yearly'], default=None,
        help='Restrict DRP aggregation to a single period tier',
    )
    parser.add_argument(
        '--limit', '-n', type=int, default=None,
        help='Stop after processing this many scenes',
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Logging goes to the directory of the config file
    config_path = Path(args.config).resolve()
    log_path = config_path.parent / 'batch_wq.log'
    _setup_logging(str(log_path))
    logger = logging.getLogger('batch_wq')

    logger.info('============================================================')
    logger.info('WQ retrieval pipeline started')
    logger.info('Config: %s', config_path)

    # ── load config ───────────────────────────────────────────────────────────
    # Add the batch/ dir to sys.path so wq_retrieve is importable
    batch_dir = str(config_path.parent)
    if batch_dir not in sys.path:
        sys.path.insert(0, batch_dir)

    from wq_retrieve.config import load_config
    cfg = load_config(config_path)

    # Inject gaac_gen if configured (enables Cinputmask for mask reading)
    if cfg.gaac_gen_dir and cfg.gaac_gen_dir not in sys.path:
        sys.path.insert(0, cfg.gaac_gen_dir)
        logger.info('gaac_gen injected: %s', cfg.gaac_gen_dir)

    # Import algorithms package to trigger all @register_algorithm decorators
    from wq_retrieve import algorithms as _  # noqa: F401
    from wq_retrieve.registry import list_algorithms
    logger.info('Registered algorithms: %s', list_algorithms())

    from wq_retrieve.processor import SceneProcessor
    from wq_retrieve.aggregator import DRPAggregator

    processor  = SceneProcessor(cfg)
    aggregator = DRPAggregator(cfg)

    # ── collect scene directories ─────────────────────────────────────────────
    l2_dir = Path(cfg.l2_dir)
    scene_dirs = sorted(d for d in l2_dir.iterdir()
                        if d.is_dir() and d.name.endswith('_GAAC'))

    if not scene_dirs:
        logger.warning('No *_GAAC directories found in %s', l2_dir)
        return 1

    logger.info('Found %d scene(s) in %s', len(scene_dirs), l2_dir)

    # ── PMP phase ─────────────────────────────────────────────────────────────
    processed_dates: set[datetime.date] = set()

    if not args.drp_only:
        count = 0
        for scene_dir in scene_dirs:
            # Filter by --scene flag
            if args.scene and not any(s in scene_dir.name for s in args.scene):
                continue

            outputs = processor.process_scene(scene_dir)

            if outputs:
                # Parse date from the scene name (tokens[2] of the _GAAC dir name)
                tokens = scene_dir.name.replace('_GAAC', '').split('_')
                try:
                    tok = tokens[2]
                    d = datetime.date(int(tok[:4]), int(tok[4:6]), int(tok[6:8]))
                    processed_dates.add(d)
                except (IndexError, ValueError):
                    pass

            count += 1
            if args.limit and count >= args.limit:
                logger.info('Reached --limit %d; stopping PMP phase', args.limit)
                break

        logger.info('PMP phase done. Processed %d scene(s), %d unique date(s)',
                    count, len(processed_dates))

    # ── DRP phase ─────────────────────────────────────────────────────────────
    if not args.pmp_only:
        cfg_periods = cfg.aggregation.get('periods', ['daily', 'monthly', 'yearly'])
        periods = [args.period] if args.period else cfg_periods

        enabled_products = {
            prod: cfg.wq_products[prod]
            for prod in cfg.wq_products
            if cfg.wq_products[prod].get('enabled', True)
        }

        if 'daily' in periods:
            # Determine which dates to aggregate
            if processed_dates:
                dates_to_agg = processed_dates
            else:
                # drp-only mode: scan all PMP directories for dates
                dates_to_agg = _scan_pmp_dates(Path(cfg.l3_dir))

            for date in sorted(dates_to_agg):
                for prod, prod_cfg in enabled_products.items():
                    aggregator.aggregate_daily(
                        date=date, product=prod,
                        algorithm=prod_cfg['algorithm'],
                        units=_get_units(prod, prod_cfg['algorithm']),
                    )

        if 'monthly' in periods:
            months_done: set[tuple[int, int]] = set()
            for date in sorted(processed_dates or _scan_pmp_dates(Path(cfg.l3_dir))):
                ym = (date.year, date.month)
                if ym not in months_done:
                    months_done.add(ym)
                    for prod, prod_cfg in enabled_products.items():
                        aggregator.aggregate_monthly(
                            year=ym[0], month=ym[1], product=prod,
                            algorithm=prod_cfg['algorithm'],
                            units=_get_units(prod, prod_cfg['algorithm']),
                        )

        if 'yearly' in periods:
            years_done: set[int] = set()
            for date in sorted(processed_dates or _scan_pmp_dates(Path(cfg.l3_dir))):
                y = date.year
                if y not in years_done:
                    years_done.add(y)
                    for prod, prod_cfg in enabled_products.items():
                        aggregator.aggregate_yearly(
                            year=y, product=prod,
                            algorithm=prod_cfg['algorithm'],
                            units=_get_units(prod, prod_cfg['algorithm']),
                        )

        logger.info('DRP phase done.')

    logger.info('Pipeline finished.')
    return 0


def _scan_pmp_dates(l3_dir: Path) -> set[datetime.date]:
    """Scan PMP directory tree for available dates (used in --drp-only mode)."""
    dates: set[datetime.date] = set()
    pmp_root = l3_dir / 'PMP'
    if not pmp_root.exists():
        return dates
    for tif in pmp_root.rglob('*.tif'):
        parts = tif.relative_to(pmp_root).parts
        # Structure: YYYY/MM/DD/<scene>/<file>.tif
        if len(parts) >= 3:
            try:
                dates.add(datetime.date(int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return dates


def _get_units(product: str, algorithm: str) -> str:
    """Look up units from the registered algorithm class."""
    try:
        from wq_retrieve.registry import get_algorithm
        algo_cls = get_algorithm(product, algorithm)
        return algo_cls.units
    except Exception:
        return ''


if __name__ == '__main__':
    sys.exit(main())
