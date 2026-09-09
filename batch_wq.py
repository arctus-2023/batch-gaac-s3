#!/usr/bin/env python3
"""Water Quality retrieval pipeline for GAAC L2 scenes.

Reads atmospherically corrected water-leaving reflectance (*_rhor_rhow.tif)
from the GAAC batch pipeline and retrieves CDOM, Chla, SPM, and Turbidity.

Supported sensors
-----------------
Sentinel-3A/B OLCI, Sentinel-2A/B/C MSI, Landsat-8 OLI, Landsat-9 OLI-2.
Each scene's instrument is read from the `sensor` tag the GAAC writer stamps on
the rhow file (falling back to the scene name).  Algorithms declare *nominal*
wavelengths and are matched to the nearest band the scene actually carries, so
one config runs across all four missions; a product may also name a different
algorithm per sensor (see `wq_products.<product>.sensors` in the config).

Products
--------
PMP (Primary): one GeoTIFF per variable per scene
DRP daily    : temporal merge of same-day PMPs (mean / std / count)
DRP monthly  : pooled aggregate of daily DRPs
DRP yearly   : pooled aggregate of monthly DRPs

DRP composites merge every contributing mission for a period. The missions do
not share a grid, so each PMP is warped onto the common target grid set by
`aggregation.grid`; `aggregation.sensors` restricts which of them take part.

Usage
-----
# Full run (PMP + all DRP tiers):
    python batch_wq.py wq_config.yml

# Single scene (substring match on directory name):
    python batch_wq.py wq_config.yml --scene S3A_L1TOA_20250615

# Only Landsat scenes (accepts S3/S3A/OLCI/S2/MSI/L8/L9/OLI/L8_OLI/…):
    python batch_wq.py wq_config.yml --sensor L8 --sensor L9

# PMP only (no aggregation):
    python batch_wq.py wq_config.yml --pmp-only

# DRP only (requires PMPs to already exist):
    python batch_wq.py wq_config.yml --drp-only

# Specific DRP period:
    python batch_wq.py wq_config.yml --period daily

# Limit number of scenes processed:
    python batch_wq.py wq_config.yml --limit 3

# List every registered algorithm with its bands and sensors, then exit:
    python batch_wq.py wq_config.yml --list-algorithms
"""

from __future__ import annotations

import argparse
import datetime
import logging
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
        description='Water quality retrieval for GAAC L2 products '
                    '(Sentinel-3 OLCI, Sentinel-2 MSI, Landsat-8/9 OLI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('config', help='Path to wq_config.yml')
    parser.add_argument(
        '--scene', '-s', action='append', default=None, metavar='STR',
        help='Process only scene dirs whose name contains STR (repeatable)',
    )
    parser.add_argument(
        '--sensor', action='append', default=None, metavar='NAME',
        help='Process only scenes from this sensor, platform, or instrument '
             'family — e.g. L8, L9, OLI, S2, MSI, S3A, OLCI (repeatable)',
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
    parser.add_argument(
        '--list-algorithms', action='store_true',
        help='Print every registered algorithm with its bands and sensors, then exit',
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
    # Make wq_retrieve importable: it sits next to this script, but historically
    # the config lived beside it too, so keep honouring the config's directory.
    for candidate in (Path(__file__).resolve().parent, config_path.parent):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    from wq_retrieve.config import load_config
    cfg = load_config(config_path)

    # Inject gaac_gen if configured (enables Cinputmask for mask reading)
    if cfg.gaac_gen_dir and cfg.gaac_gen_dir not in sys.path:
        sys.path.insert(0, cfg.gaac_gen_dir)
        logger.info('gaac_gen injected: %s', cfg.gaac_gen_dir)

    # Import algorithms package to trigger all @register_algorithm decorators
    from wq_retrieve import algorithms as _  # noqa: F401
    from wq_retrieve.registry import list_algorithms
    from wq_retrieve.scene import parse_scene_date

    if args.list_algorithms:
        _print_algorithms()
        return 0

    logger.info('Registered algorithms: %s', list_algorithms())
    logger.info('Band matching tolerance: ±%d nm', cfg.band_tolerance)

    from wq_retrieve.processor import SceneProcessor
    from wq_retrieve.aggregator import DRPAggregator

    processor  = SceneProcessor(cfg)
    aggregator = DRPAggregator(cfg)

    # ── collect scene directories ─────────────────────────────────────────────
    # rglob, not iterdir: scenes are often grouped in per-mission subdirectories
    # (e.g. L2/S2/<scene>_GAAC alongside L2/<LC08 scene>_GAAC).
    l2_dir = Path(cfg.l2_dir)
    scene_dirs = sorted(d for d in l2_dir.rglob('*_GAAC') if d.is_dir())

    if not scene_dirs:
        logger.warning('No *_GAAC directories found under %s', l2_dir)
        return 1

    logger.info('Found %d scene(s) under %s', len(scene_dirs), l2_dir)

    # ── PMP phase ─────────────────────────────────────────────────────────────
    # dates that produced output, to drive aggregation
    processed: set[datetime.date] = set()

    if not args.drp_only:
        count = 0
        for scene_dir in scene_dirs:
            # Filter by --scene flag
            if args.scene and not any(s in scene_dir.name for s in args.scene):
                continue
            if args.sensor and not _sensor_matches(scene_dir.name, args.sensor):
                continue

            outputs = processor.process_scene(scene_dir)

            if outputs:
                date = parse_scene_date(scene_dir.name)
                if date:
                    processed.add(date)

            count += 1
            if args.limit and count >= args.limit:
                logger.info('Reached --limit %d; stopping PMP phase', args.limit)
                break

        logger.info('PMP phase done. Processed %d scene(s) over %d date(s)',
                    count, len(processed))

    # ── DRP phase ─────────────────────────────────────────────────────────────
    if not args.pmp_only:
        cfg_periods = cfg.aggregation.get('periods', ['daily', 'monthly', 'yearly'])
        periods = [args.period] if args.period else cfg_periods

        # --sensor narrows the configured DRP sensor set for this run
        if args.sensor:
            aggregator.sensors = args.sensor
        logger.info('DRP sensors: %s',
                    ','.join(aggregator.sensors) if aggregator.sensors else 'all')
        if cfg.drp_grid:
            g = cfg.drp_grid
            logger.info('DRP grid: %s  %s m  bounds=%s  resampling=%s',
                        g.crs, g.resolution[0],
                        'auto (union of inputs)' if g.is_auto_bounds else list(g.bounds),
                        g.resampling)
        else:
            logger.info('DRP grid: not configured — derived from the inputs')

        # drp-only mode: rebuild the date index by scanning the PMP tree
        dates = processed or _scan_pmp_dates(Path(cfg.l3_dir))
        if not dates:
            logger.warning('No PMP products found to aggregate under %s', cfg.l3_dir)

        products = list(cfg.wq_products)

        if 'daily' in periods:
            for date in sorted(dates):
                for prod in products:
                    aggregator.aggregate_daily(date=date, product=prod)

        if 'monthly' in periods:
            for year, month in sorted({(d.year, d.month) for d in dates}):
                for prod in products:
                    aggregator.aggregate_monthly(year=year, month=month, product=prod)

        if 'yearly' in periods:
            for year in sorted({d.year for d in dates}):
                for prod in products:
                    aggregator.aggregate_yearly(year=year, product=prod)

        logger.info('DRP phase done.')

    logger.info('Pipeline finished.')
    return 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _sensor_matches(scene_name: str, wanted: list[str]) -> bool:
    """True when a scene name's sensor matches any --sensor selector."""
    from wq_retrieve.sensors import matches, sensor_from_name
    spec = sensor_from_name(scene_name)
    return spec is not None and matches(spec, wanted)


def _scan_pmp_dates(l3_dir: Path) -> set[datetime.date]:
    """Scan the PMP tree for the dates that have products.

    Used in --drp-only mode, where no scenes were processed this run.
    """
    dates: set[datetime.date] = set()
    pmp_root = l3_dir / 'PMP'
    if not pmp_root.exists():
        return dates
    for tif in pmp_root.rglob('*.tif'):
        parts = tif.relative_to(pmp_root).parts
        # Structure: YYYY/MM/DD/<scene>/<file>.tif
        if len(parts) < 4:
            continue
        try:
            dates.add(datetime.date(int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return dates


def _print_algorithms() -> None:
    """Print the algorithm registry as a table: product, name, sensors, bands."""
    from wq_retrieve.registry import get_algorithm, list_algorithms
    from wq_retrieve.sensors import FAMILIES

    rows = []
    for product, names in list_algorithms().items():
        for name in names:
            cls = get_algorithm(product, name)
            fams = cls.families or FAMILIES
            bands = {f: cls.bands_for(f) for f in fams}
            same = len({tuple(b) for b in bands.values()}) == 1
            band_txt = (','.join(str(b) for b in next(iter(bands.values())))
                        if same else
                        '  '.join(f'{f}:{",".join(str(b) for b in bs)}'
                                  for f, bs in bands.items()))
            rows.append((product, name, '/'.join(fams), cls.input_quantity, band_txt))

    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    header = ('product', 'algorithm', 'sensors', 'input', 'nominal bands [nm]')
    widths = [max(w, len(h)) for w, h in zip(widths, header)]
    fmt = '  '.join(f'{{:<{w}}}' for w in widths) + '  {}'
    print(fmt.format(*header))
    print('  '.join('-' * w for w in widths) + '  ' + '-' * 20)
    for row in rows:
        print(fmt.format(*row))


if __name__ == '__main__':
    sys.exit(main())
