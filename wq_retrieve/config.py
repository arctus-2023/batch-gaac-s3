"""WQConfig — configuration dataclass and YAML loader."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .sensors import DEFAULT_TOLERANCE, SensorSpec


@dataclass(frozen=True)
class DRPGrid:
    """Target grid every PMP is warped onto before DRP compositing.

    Fixing the grid in the config keeps composites comparable across runs even
    as the set of available scenes changes. `bounds` may be the string 'auto',
    in which case the union of the contributing PMPs is used instead — handy
    for exploration, but the extent then moves whenever the scene set does.
    """
    crs: str
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float] | None   # None = 'auto' (union)
    resampling: str = 'average'

    @property
    def is_auto_bounds(self) -> bool:
        return self.bounds is None


def _parse_grid(raw: dict | None) -> DRPGrid | None:
    """Build a DRPGrid from the `aggregation.grid` block, or None if absent."""
    if not raw:
        return None
    missing = [k for k in ('crs', 'resolution') if k not in raw]
    if missing:
        raise ValueError(f'aggregation.grid is missing required key(s): {missing}')

    res = raw['resolution']
    res = (float(res), float(res)) if isinstance(res, (int, float)) else (
        float(res[0]), float(res[1]))
    if res[0] <= 0 or res[1] <= 0:
        raise ValueError(f'aggregation.grid.resolution must be positive, got {res}')

    bounds = raw.get('bounds', 'auto')
    if isinstance(bounds, str):
        if bounds.lower() != 'auto':
            raise ValueError(
                f"aggregation.grid.bounds must be 'auto' or "
                f'[minx, miny, maxx, maxy], got {bounds!r}'
            )
        bounds = None
    else:
        if len(bounds) != 4:
            raise ValueError(
                f'aggregation.grid.bounds needs 4 values [minx, miny, maxx, maxy], '
                f'got {len(bounds)}'
            )
        bounds = tuple(float(b) for b in bounds)
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError(
                f'aggregation.grid.bounds must satisfy minx<maxx and miny<maxy, '
                f'got {bounds}'
            )

    return DRPGrid(
        crs=str(raw['crs']),
        resolution=res,
        bounds=bounds,
        resampling=str(raw.get('resampling', 'average')),
    )


_OUTLIER_DEFAULT_THRESHOLD = {'mad': 3.5, 'iqr': 1.5}


@dataclass(frozen=True)
class OutlierConfig:
    """Per-pixel outlier removal applied before merging monthly and yearly DRPs.

    method      : 'mad'  -- modified z-score, |0.6745 (x - median) / MAD| > threshold
                            (Iglewicz & Hoaglin; 3.5 is their recommendation)
                  'iqr'  -- outside [Q1 - threshold*IQR, Q3 + threshold*IQR]
                            (Tukey; 1.5 is the usual fence)
                  'none' -- keep every value
    threshold   : see method; defaults to 3.5 for mad, 1.5 for iqr
    min_samples : a pixel with fewer contributing values is left alone -- two or
                  three numbers cannot say which of them is the outlier
    spatial_clip: also apply the old per-file 5-95 % percentile trim. Off by
                  default at these tiers: it removes the tails of every image
                  whether or not they are outliers, and applied again at each
                  tier it compounds -- the daily tier has already trimmed once.
    """
    method: str = 'mad'
    threshold: float = 3.5
    min_samples: int = 3
    spatial_clip: bool = False


def _parse_outliers(raw: dict | None) -> OutlierConfig:
    raw = raw or {}
    method = str(raw.get('method', 'mad')).lower()
    if method not in ('mad', 'iqr', 'none'):
        raise ValueError(f"aggregation.outliers.method must be mad, iqr or none, got {method!r}")
    threshold = float(raw.get('threshold', _OUTLIER_DEFAULT_THRESHOLD.get(method, 0.0)))
    if method != 'none' and threshold <= 0:
        raise ValueError(f'aggregation.outliers.threshold must be positive, got {threshold}')
    min_samples = int(raw.get('min_samples', 3))
    if min_samples < 3:
        raise ValueError('aggregation.outliers.min_samples must be at least 3; '
                         f'got {min_samples}')
    return OutlierConfig(method=method, threshold=threshold, min_samples=min_samples,
                         spatial_clip=bool(raw.get('spatial_clip', False)))


@dataclass
class WQConfig:
    l2_dir: str
    l3_dir: str
    aoi_name: str
    wq_products: dict[str, dict[str, Any]]
    aggregation: dict[str, Any]
    gaac_gen_dir: str | None = None
    replace_output: bool = False
    #: max |nominal − actual| accepted when matching algorithm bands to a scene [nm]
    band_tolerance: int = DEFAULT_TOLERANCE
    #: sensor selectors admitted into DRP composites; empty/None = every sensor
    drp_sensors: list[str] | None = None
    #: common grid PMPs are warped onto before compositing; None = derive it
    drp_grid: DRPGrid | None = None
    #: flag-band bits that exclude a pixel from the PMPs (names, numbers, opt_exclude)
    mask_exclude_bits: Any = 'opt_exclude'
    #: per-pixel temporal outlier removal for monthly/yearly DRPs
    outliers: 'OutlierConfig | None' = None


def _normalize_variant(prod_name: str, key: str, raw: Any) -> dict[str, Any]:
    """Normalize one product variant (the default or a per-sensor override).

    Accepts either a bare algorithm name (``chla: {S2: ndci}``) or a mapping
    (``chla: {S2: {algorithm: ndci, params: {...}}}``).
    """
    if isinstance(raw, str):
        return {'enabled': True, 'algorithm': raw, 'params': {}}
    if not isinstance(raw, dict):
        raise ValueError(
            f'wq_products.{prod_name}.{key} must be a string or a mapping, '
            f'got {type(raw).__name__}'
        )
    if 'algorithm' not in raw:
        raise ValueError(
            f'wq_products.{prod_name}.{key} is missing required key "algorithm"'
        )
    return {
        'enabled':   bool(raw.get('enabled', True)),
        'algorithm': raw['algorithm'],
        'params':    raw.get('params') or {},
    }


def resolve_product(prod_cfg: dict[str, Any], sensor: SensorSpec) -> dict[str, Any] | None:
    """Pick the algorithm variant to use for `sensor`, or None if disabled.

    Sensor overrides are matched most-specific-first: the exact sensor tag
    (``L8_OLI``), then the platform label (``S3A``, ``S2``, ``L8``), then the
    instrument family (``OLCI``, ``MSI``, ``OLI``).  With no match the product's
    top-level algorithm applies.
    """
    overrides: dict[str, dict] = prod_cfg.get('sensors') or {}
    for key in (sensor.key, sensor.label, sensor.family):
        variant = overrides.get(key.upper())
        if variant is not None:
            return variant if variant['enabled'] else None
    return prod_cfg if prod_cfg.get('enabled', True) else None


def load_config(path: str | Path) -> WQConfig:
    """Load and validate a WQ pipeline YAML config file."""
    with open(path) as f:
        raw: dict = yaml.safe_load(f)

    required = ['l2_dir', 'l3_dir', 'aoi_name', 'wq_products']
    for key in required:
        if key not in raw:
            raise ValueError(f'Config missing required key: {key!r}')

    # Normalize each product entry
    products: dict[str, dict[str, Any]] = {}
    for prod_name, prod_cfg in raw['wq_products'].items():
        if not isinstance(prod_cfg, dict):
            raise ValueError(
                f'wq_products.{prod_name} must be a dict, got {type(prod_cfg)}'
            )
        entry = _normalize_variant(prod_name, 'algorithm', prod_cfg)
        entry['sensors'] = {
            str(sensor_key).upper(): _normalize_variant(prod_name, sensor_key, variant)
            for sensor_key, variant in (prod_cfg.get('sensors') or {}).items()
        }
        products[prod_name] = entry

    aggregation = raw.get('aggregation', {
        'method':  'mean',
        'periods': ['daily', 'monthly', 'yearly'],
    })

    drp_sensors = aggregation.get('sensors')
    if isinstance(drp_sensors, str):
        drp_sensors = None if drp_sensors.lower() == 'all' else [drp_sensors]
    elif drp_sensors is not None:
        drp_sensors = [str(x) for x in drp_sensors]
        if any(x.lower() == 'all' for x in drp_sensors):
            drp_sensors = None

    return WQConfig(
        l2_dir=str(raw['l2_dir']),
        l3_dir=str(raw['l3_dir']),
        aoi_name=str(raw['aoi_name']),
        wq_products=products,
        aggregation=aggregation,
        gaac_gen_dir=raw.get('gaac_gen_dir'),
        replace_output=bool(raw.get('replace_output', False)),
        band_tolerance=int(raw.get('band_tolerance', DEFAULT_TOLERANCE)),
        drp_sensors=drp_sensors,
        drp_grid=_parse_grid(aggregation.get('grid')),
        mask_exclude_bits=(raw.get('mask') or {}).get('exclude_bits', 'opt_exclude'),
        outliers=_parse_outliers(aggregation.get('outliers')),
    )
