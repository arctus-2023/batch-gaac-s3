"""DRPAggregator — temporal compositing from PMP TIFs to daily / monthly / yearly DRPs.

Aggregation strategy
--------------------
PMP → daily  : Welford online algorithm (memory-efficient mean + variance
                across all scenes for a given calendar day)
daily → monthly: count-weighted pooled mean and variance across daily DRPs
monthly → yearly: same pooled weighting

Sensor merging
--------------
Composites merge every contributing mission — OLCI, MSI and OLI scenes for the
same period land in one product. Because the missions do not share a grid, each
input is warped onto a common target grid first (`aggregation.grid` in the
config; see `_resolve_grid` for what happens when it is omitted). Restrict which
sensors take part with `aggregation.sensors`.

Outlier rejection
-----------------
Before each array enters the accumulator, values outside the [5 %, 95 %]
percentile of that array's finite pixels are set to NaN and excluded from
the mean, std, and count.  The percentile bounds are computed per-file, at the
file's native resolution and before warping, so that scenes with very different
dynamic ranges are treated independently and extreme pixels cannot leak into
the average of a coarser target cell.

Each DRP TIF has 3 bands: mean, std, count.
"""

from __future__ import annotations
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

from .config import DRPGrid, WQConfig
from .io import write_drp_tif
from .sensors import SensorSpec, detect_sensor, matches

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Grid:
    """A concrete target grid: what every input is warped onto."""
    crs: CRS
    transform: object
    width: int
    height: int
    resampling: Resampling

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def matches(self, crs, transform, shape) -> bool:
        """True when a source is already on this grid, so no warp is needed."""
        return (crs == self.crs and transform == self.transform
                and tuple(shape) == self.shape)

    def describe(self) -> str:
        return (f'{self.crs.to_string()}  {self.transform.a:g} m  '
                f'{self.width}x{self.height}  ({self.resampling.name})')


def _resampling(name: str) -> Resampling:
    try:
        return Resampling[name]
    except KeyError:
        raise ValueError(
            f'Unknown resampling {name!r}; choose from '
            f'{[r.name for r in Resampling if r.value <= 7]}'
        ) from None


class DRPAggregator:
    """Aggregate PMP TIFs into daily, monthly, and yearly DRP composites."""

    def __init__(self, cfg: WQConfig) -> None:
        self.cfg = cfg
        self.l3 = Path(cfg.l3_dir)
        #: sensor selectors admitted into composites; None = every sensor
        self.sensors = cfg.drp_sensors

    # ── grid resolution ───────────────────────────────────────────────────────

    def _resolve_grid(self, paths: list[Path]) -> _Grid | None:
        """Decide the target grid for a set of inputs.

        With `aggregation.grid` configured, that grid is used verbatim (its
        bounds may be 'auto', i.e. the union of the inputs).  Without it, inputs
        that already share one grid are composited in place — no warping, no
        loss — and a mixed set falls back to the finest contributing resolution
        with a warning, since an implicit grid moves whenever the scene set does.
        """
        specs = []
        for p in paths:
            try:
                with rasterio.open(p) as src:
                    specs.append((src.crs, src.transform, src.shape, src.bounds))
            except Exception as exc:
                logger.warning('Could not read %s: %s', p, exc)
        if not specs:
            return None

        cfg_grid: DRPGrid | None = self.cfg.drp_grid
        if cfg_grid is not None:
            crs = CRS.from_string(cfg_grid.crs)
            xres, yres = cfg_grid.resolution
            bounds = cfg_grid.bounds or self._union_bounds(specs, crs)
            minx, miny, maxx, maxy = bounds
            width  = max(1, int(round((maxx - minx) / xres)))
            height = max(1, int(round((maxy - miny) / yres)))
            return _Grid(crs, from_origin(minx, maxy, xres, yres), width, height,
                         _resampling(cfg_grid.resampling))

        # No configured grid: if everything already agrees, keep it as-is.
        first = specs[0]
        if all((s[0], s[1], s[2]) == (first[0], first[1], first[2]) for s in specs):
            return _Grid(first[0], first[1], first[2][1], first[2][0],
                         Resampling.average)

        finest = min(specs, key=lambda s: (abs(s[1].a), str(s[0])))
        crs, xres, yres = finest[0], abs(finest[1].a), abs(finest[1].e)
        minx, miny, maxx, maxy = self._union_bounds(specs, crs)
        width  = max(1, int(round((maxx - minx) / xres)))
        height = max(1, int(round((maxy - miny) / yres)))
        logger.warning(
            'Inputs span %d different grids and aggregation.grid is not set; '
            'falling back to the finest (%s, %g m, %dx%d). Set aggregation.grid '
            'to keep composites comparable as the scene set changes.',
            len({(s[0], s[1], s[2]) for s in specs}), crs.to_string(),
            xres, width, height,
        )
        return _Grid(crs, from_origin(minx, maxy, xres, yres), width, height,
                     Resampling.average)

    @staticmethod
    def _union_bounds(specs: list[tuple], crs: CRS) -> tuple[float, float, float, float]:
        """Union of input bounds, expressed in `crs`."""
        boxes = []
        for src_crs, _, _, b in specs:
            boxes.append(b if src_crs == crs
                         else transform_bounds(src_crs, crs, *b, densify_pts=21))
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    @staticmethod
    def _read_aligned(
        path: Path, band: int, grid: _Grid, clip: bool = False,
        resampling: Resampling | None = None,
    ) -> np.ndarray | None:
        """Read one band onto `grid`, optionally clipping outliers beforehand.

        Clipping happens at native resolution, before any warp, so that extreme
        pixels are removed rather than smeared into coarser target cells.
        """
        try:
            with rasterio.open(path) as src:
                arr = src.read(band).astype(np.float32)
                src_crs, src_transform, src_shape = src.crs, src.transform, src.shape
        except Exception as exc:
            logger.warning('Could not read %s: %s', path, exc)
            return None

        if clip:
            arr = DRPAggregator._clip_outliers(arr)

        if grid.matches(src_crs, src_transform, src_shape):
            return arr

        dst = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=arr, destination=dst,
            src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
            dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
            resampling=resampling or grid.resampling,
        )
        return dst

    # ── daily ─────────────────────────────────────────────────────────────────

    def aggregate_daily(
        self,
        date: datetime.date,
        product: str,
        algorithm: str = '',
        units: str = '',
    ) -> Path | None:
        """Aggregate all PMPs for (date, product), across every admitted sensor.

        Returns the output path, or None if no PMP files were found.
        """
        pmp_paths = self._find_pmp(date, product)
        if not pmp_paths:
            logger.debug('No PMPs for %s on %s', product, date)
            return None

        out_path = self._drp_daily_path(date, product)
        if out_path.exists() and not self.cfg.replace_output:
            logger.debug('Daily DRP exists, skip: %s', out_path.name)
            return out_path

        grid = self._resolve_grid(pmp_paths)
        if grid is None:
            return None

        mean, std, count = self._welford_stack(pmp_paths, grid)
        if mean is None:
            return None

        prov = self._provenance(pmp_paths)
        algorithm = algorithm or prov['algorithms']
        units = units or prov['units']

        write_drp_tif(
            path=out_path,
            mean=mean, std=std, count=count,
            meta=self._grid_meta(grid),
            product=product,
            algorithm=algorithm,
            units=units,
            period='daily',
            date_label=date.strftime('%Y%m%d'),
            extra_tags={'sensors': prov['sensors'], 'n_scenes': str(len(pmp_paths)),
                        'grid': grid.describe()},
        )
        n_valid = int(np.isfinite(mean).sum())
        logger.info('Daily DRP  %-10s %s  scenes=%d [%s]  valid_px=%d',
                    product, date, len(pmp_paths), prov['sensors'], n_valid)
        return out_path

    # ── monthly ────────────────────────────────────────────────────────────────

    def aggregate_monthly(
        self,
        year: int,
        month: int,
        product: str,
        algorithm: str = '',
        units: str = '',
    ) -> Path | None:
        """Aggregate all daily DRPs for (year, month, product) → monthly DRP."""
        daily_paths = sorted(
            (self.l3 / 'DRP' / 'daily' / f'{year:04d}' / f'{month:02d}').rglob(
                f'{self.cfg.aoi_name}_*_{product}.tif'
            )
        )
        if not daily_paths:
            return None

        out_path = self._drp_monthly_path(year, month, product)
        if out_path.exists() and not self.cfg.replace_output:
            return out_path

        grid = self._resolve_grid(daily_paths)
        if grid is None:
            return None

        mean, std, count = self._pooled_stack(daily_paths, grid)
        if mean is None:
            return None

        prov = self._provenance(daily_paths)
        algorithm = algorithm or prov['algorithms']
        units = units or prov['units']
        write_drp_tif(
            path=out_path,
            mean=mean, std=std, count=count,
            meta=self._grid_meta(grid),
            product=product,
            algorithm=algorithm,
            units=units,
            period='monthly',
            date_label=f'{year:04d}{month:02d}',
            extra_tags={'n_days': str(len(daily_paths)), 'grid': grid.describe(),
                        'sensors': prov['sensors']},
        )
        logger.info('Monthly DRP  %-10s %04d-%02d  days=%d', product, year, month,
                    len(daily_paths))
        return out_path

    # ── yearly ─────────────────────────────────────────────────────────────────

    def aggregate_yearly(
        self,
        year: int,
        product: str,
        algorithm: str = '',
        units: str = '',
    ) -> Path | None:
        """Aggregate all monthly DRPs for (year, product) → yearly DRP."""
        monthly_paths = sorted(
            (self.l3 / 'DRP' / 'monthly' / f'{year:04d}').rglob(
                f'{self.cfg.aoi_name}_*_{product}.tif'
            )
        )
        if not monthly_paths:
            return None

        out_path = self._drp_yearly_path(year, product)
        if out_path.exists() and not self.cfg.replace_output:
            return out_path

        grid = self._resolve_grid(monthly_paths)
        if grid is None:
            return None

        mean, std, count = self._pooled_stack(monthly_paths, grid)
        if mean is None:
            return None

        prov = self._provenance(monthly_paths)
        algorithm = algorithm or prov['algorithms']
        units = units or prov['units']
        write_drp_tif(
            path=out_path,
            mean=mean, std=std, count=count,
            meta=self._grid_meta(grid),
            product=product,
            algorithm=algorithm,
            units=units,
            period='yearly',
            date_label=f'{year:04d}',
            extra_tags={'n_months': str(len(monthly_paths)), 'grid': grid.describe(),
                        'sensors': prov['sensors']},
        )
        logger.info('Yearly DRP  %-10s %04d  months=%d', product, year,
                    len(monthly_paths))
        return out_path

    # ── core statistics ────────────────────────────────────────────────────────

    @staticmethod
    def _clip_outliers(arr: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
        """Return a copy of arr with values outside [lo, hi] percentile set to NaN.

        Percentiles are computed from finite (non-NaN) values only.  If fewer
        than 2 finite values exist the array is returned unchanged.
        """
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            return arr.copy()
        p_lo, p_hi = np.percentile(finite, [lo, hi])
        result = arr.copy()
        result[(arr < p_lo) | (arr > p_hi)] = np.nan
        return result

    def _welford_stack(
        self, paths: list[Path], grid: _Grid
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Compute mean, std, count across single-band TIFs using Welford's algorithm.

        Processes one file at a time → O(H×W) memory regardless of file count.
        Each input is outlier-clipped at native resolution, then warped onto
        `grid`, so scenes from different missions accumulate together.
        """
        H, W = grid.shape
        n    = np.zeros((H, W), dtype=np.float32)
        mean = np.zeros((H, W), dtype=np.float32)
        M2   = np.zeros((H, W), dtype=np.float32)
        used = 0

        for p in paths:
            arr = self._read_aligned(p, 1, grid, clip=True)
            if arr is None:
                continue
            used += 1

            valid = np.isfinite(arr)
            n += valid.astype(np.float32)
            safe_n = np.where(n > 0, n, 1.0)
            delta  = np.where(valid, arr - mean, 0.0)
            mean  += np.where(valid, delta / safe_n, 0.0)
            delta2 = np.where(valid, arr - mean, 0.0)
            M2    += delta * delta2

        if used == 0:
            return None, None, None

        std = np.where(n > 1, np.sqrt(M2 / np.where(n > 1, n - 1, 1.0)), 0.0)
        mean = np.where(n > 0, mean, np.nan)
        std  = np.where(n > 0, std,  np.nan)
        return mean, std, n

    def _pooled_stack(
        self, drp_paths: list[Path], grid: _Grid
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Combine DRP 3-band files (mean/std/count) using count-weighted pooling.

        For each pixel:
          total_count = Σ count_i
          pooled_mean = Σ (mean_i · count_i) / total_count
          pooled_var  = Σ count_i · (var_i + (mean_i − pooled_mean)²) / total_count

        Mean values outside the [5 %, 95 %] percentile of each DRP file are
        excluded; their associated counts are zeroed so they do not contribute
        to the pooled total.
        """
        H, W = grid.shape
        total_count = np.zeros((H, W), dtype=np.float32)
        mean_acc    = np.zeros((H, W), dtype=np.float32)
        used = 0

        # Pass 1: compute pooled mean
        for p in drp_paths:
            m = self._read_aligned(p, 1, grid, clip=True)
            c = self._read_aligned(p, 3, grid)
            if m is None or c is None:
                continue
            used += 1
            # zero count where m is NaN (original nodata or clipped outlier)
            c = np.where(np.isfinite(c) & np.isfinite(m), c, 0.0)
            m = np.where(np.isfinite(m), m, 0.0)
            total_count += c
            mean_acc    += m * c

        if used == 0:
            return None, None, None

        safe_count = np.where(total_count > 0, total_count, 1.0)
        pooled_mean = mean_acc / safe_count

        # Pass 2: compute pooled variance (same outlier mask as pass 1)
        var_acc = np.zeros_like(pooled_mean)
        for p in drp_paths:
            m = self._read_aligned(p, 1, grid, clip=True)
            s = self._read_aligned(p, 2, grid)
            c = self._read_aligned(p, 3, grid)
            if m is None or s is None or c is None:
                continue
            c = np.where(np.isfinite(c) & np.isfinite(m), c, 0.0)
            m = np.where(np.isfinite(m), m, 0.0)
            s = np.where(np.isfinite(s), s, 0.0)
            var_acc += c * (s**2 + (m - pooled_mean)**2)

        pooled_var = var_acc / safe_count
        pooled_std = np.sqrt(np.where(pooled_var > 0, pooled_var, 0.0))

        pooled_mean = np.where(total_count > 0, pooled_mean, np.nan)
        pooled_std  = np.where(total_count > 0, pooled_std,  np.nan)
        return pooled_mean, pooled_std, total_count

    # ── path helpers ──────────────────────────────────────────────────────────

    def _find_pmp(self, date: datetime.date, product: str) -> list[Path]:
        """PMP TIFs for (date, product) from every sensor the config admits."""
        pmp_dir = (
            self.l3 / 'PMP'
            / f'{date.year:04d}' / f'{date.month:02d}' / f'{date.day:02d}'
        )
        if not pmp_dir.exists():
            return []
        keep = []
        for p in sorted(pmp_dir.rglob(f'*_{product}.tif')):
            spec = pmp_sensor(p)
            if matches(spec, self.sensors):
                keep.append(p)
            else:
                logger.debug('Excluded by aggregation.sensors: %s (%s)',
                             p.name, spec.key)
        return keep

    def _drp_daily_path(self, date: datetime.date, product: str) -> Path:
        label = date.strftime('%Y%m%d')
        return (
            self.l3 / 'DRP' / 'daily'
            / f'{date.year:04d}' / f'{date.month:02d}' / f'{date.day:02d}'
            / f'{self.cfg.aoi_name}_{label}_{product}.tif'
        )

    def _drp_monthly_path(self, year: int, month: int, product: str) -> Path:
        label = f'{year:04d}{month:02d}'
        return (
            self.l3 / 'DRP' / 'monthly'
            / f'{year:04d}' / f'{month:02d}'
            / f'{self.cfg.aoi_name}_{label}_{product}.tif'
        )

    def _drp_yearly_path(self, year: int, product: str) -> Path:
        return (
            self.l3 / 'DRP' / 'yearly'
            / f'{year:04d}'
            / f'{self.cfg.aoi_name}_{year:04d}_{product}.tif'
        )

    @staticmethod
    def _grid_meta(grid: _Grid) -> dict:
        """rasterio profile for a DRP written on `grid`."""
        return {
            'driver': 'GTiff', 'count': 1, 'dtype': 'float32',
            'nodata': float('nan'), 'crs': grid.crs, 'transform': grid.transform,
            'width': grid.width, 'height': grid.height,
        }

    @staticmethod
    def _provenance(paths: list[Path]) -> dict[str, str]:
        """Distinct sensors, algorithms and units across the contributing files.

        A composite that merges missions may also merge algorithms — a product
        can use one algorithm on MSI and another on OLI — so the set is recorded
        in the output tags and flagged here rather than silently collapsed.
        """
        sensors, algorithms, units = [], [], []
        for p in paths:
            try:
                with rasterio.open(p) as src:
                    tags = src.tags()
            except Exception:
                continue
            # PMPs tag a single `sensor`; DRPs carry the comma-joined
            # `sensors` of everything that already went into them.
            for v in (tags.get('sensors') or tags.get('sensor') or '').split(','):
                if v and v not in sensors:
                    sensors.append(v)
            for key, acc in (('algorithm', algorithms), ('units', units)):
                for v in (tags.get(key) or '').split(','):
                    if v and v not in acc:
                        acc.append(v)
        if len(algorithms) > 1:
            logger.warning(
                'Composite merges %d different algorithms (%s) — values from '
                'different retrievals are being averaged together',
                len(algorithms), ', '.join(algorithms),
            )
        return {'sensors': ','.join(sensors), 'algorithms': ','.join(algorithms),
                'units': units[0] if units else ''}

def pmp_sensor(path: Path) -> SensorSpec:
    """Sensor of a PMP TIF, from its `sensor` tag or its filename."""
    tag = None
    try:
        with rasterio.open(path) as src:
            tag = src.tags().get('sensor')
    except Exception:
        pass
    return detect_sensor(tag, path.name)
