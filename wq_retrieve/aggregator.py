"""DRPAggregator — temporal compositing from PMP TIFs to daily / monthly / yearly DRPs.

Aggregation strategy
--------------------
PMP → daily  : Welford online algorithm (memory-efficient mean + variance
                across all scenes for a given calendar day)
daily → monthly: count-weighted pooled mean and variance across daily DRPs
monthly → yearly: same pooled weighting

Outlier rejection
-----------------
Before each array enters the accumulator, values outside the [5 %, 95 %]
percentile of that array's finite pixels are set to NaN and excluded from
the mean, std, and count.  The percentile bounds are computed per-file so
that scenes with very different dynamic ranges are treated independently.

Each DRP TIF has 3 bands: mean, std, count.
"""

from __future__ import annotations
import datetime
import logging
from pathlib import Path

import numpy as np
import rasterio

from .config import WQConfig
from .io import write_drp_tif

logger = logging.getLogger(__name__)


class DRPAggregator:
    """Aggregate PMP TIFs into daily, monthly, and yearly DRP composites."""

    def __init__(self, cfg: WQConfig) -> None:
        self.cfg = cfg
        self.l3 = Path(cfg.l3_dir)

    # ── daily ─────────────────────────────────────────────────────────────────

    def aggregate_daily(
        self,
        date: datetime.date,
        product: str,
        algorithm: str = '',
        units: str = '',
    ) -> Path | None:
        """Aggregate all PMPs for (date, product) → daily DRP TIF.

        Returns the output path, or None if no PMP files were found.
        """
        pmp_paths = self._find_pmp(date, product)
        if not pmp_paths:
            logger.debug('No PMPs for %s / %s on %s', product, date, self.cfg.aoi_name)
            return None

        out_path = self._drp_daily_path(date, product)
        if out_path.exists() and not self.cfg.replace_output:
            logger.debug('Daily DRP exists, skip: %s', out_path.name)
            return out_path

        mean, std, count = self._welford_stack(pmp_paths)
        if mean is None:
            return None

        meta = self._read_meta(pmp_paths[0])
        write_drp_tif(
            path=out_path,
            mean=mean, std=std, count=count,
            meta=meta,
            product=product,
            algorithm=algorithm,
            units=units,
            period='daily',
            date_label=date.strftime('%Y%m%d'),
        )
        n_valid = int(np.isfinite(mean).sum())
        logger.info('Daily DRP  %s  %s  valid_px=%d', product, date, n_valid)
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

        mean, std, count = self._pooled_stack(daily_paths)
        if mean is None:
            return None

        meta = self._read_meta(daily_paths[0], band=1)
        write_drp_tif(
            path=out_path,
            mean=mean, std=std, count=count,
            meta=meta,
            product=product,
            algorithm=algorithm,
            units=units,
            period='monthly',
            date_label=f'{year:04d}{month:02d}',
        )
        logger.info('Monthly DRP  %s  %04d-%02d', product, year, month)
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

        mean, std, count = self._pooled_stack(monthly_paths)
        if mean is None:
            return None

        meta = self._read_meta(monthly_paths[0], band=1)
        write_drp_tif(
            path=out_path,
            mean=mean, std=std, count=count,
            meta=meta,
            product=product,
            algorithm=algorithm,
            units=units,
            period='yearly',
            date_label=f'{year:04d}',
        )
        logger.info('Yearly DRP  %s  %04d', product, year)
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
        self, paths: list[Path]
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Compute mean, std, count across single-band TIFs using Welford's algorithm.

        Processes one file at a time → O(H×W) memory regardless of file count.
        Values outside the [5 %, 95 %] percentile of each file's finite pixels
        are excluded before accumulation.
        """
        n = mean = M2 = None

        for p in paths:
            try:
                with rasterio.open(p) as src:
                    arr = src.read(1).astype(np.float32)
            except Exception as exc:
                logger.warning('Could not read %s: %s', p, exc)
                continue

            arr = self._clip_outliers(arr)

            if n is None:
                H, W = arr.shape
                n    = np.zeros((H, W), dtype=np.float32)
                mean = np.zeros((H, W), dtype=np.float32)
                M2   = np.zeros((H, W), dtype=np.float32)
            elif arr.shape != (H, W):
                logger.warning(
                    'Shape mismatch in daily stack: expected (%d,%d), got %s for %s — skip',
                    H, W, arr.shape, p.name,
                )
                continue

            valid = np.isfinite(arr)
            n += valid.astype(np.float32)
            safe_n = np.where(n > 0, n, 1.0)
            delta  = np.where(valid, arr - mean, 0.0)
            mean  += np.where(valid, delta / safe_n, 0.0)
            delta2 = np.where(valid, arr - mean, 0.0)
            M2    += delta * delta2

        if n is None:
            return None, None, None

        std = np.where(n > 1, np.sqrt(M2 / np.where(n > 1, n - 1, 1.0)), 0.0)
        mean = np.where(n > 0, mean, np.nan)
        std  = np.where(n > 0, std,  np.nan)
        return mean, std, n

    def _pooled_stack(
        self, drp_paths: list[Path]
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
        total_count = mean_acc = None

        # Pass 1: compute pooled mean
        for p in drp_paths:
            try:
                with rasterio.open(p) as src:
                    m = src.read(1).astype(np.float32)   # mean band
                    c = src.read(3).astype(np.float32)   # count band
            except Exception as exc:
                logger.warning('Could not read DRP %s: %s', p, exc)
                continue
            m = self._clip_outliers(m)
            # zero count where m is NaN (original nodata or clipped outlier)
            c = np.where(np.isfinite(c) & np.isfinite(m), c, 0.0)
            m = np.where(np.isfinite(m), m, 0.0)
            if total_count is None:
                total_count = np.zeros_like(c)
                mean_acc    = np.zeros_like(c)
            elif c.shape != total_count.shape:
                logger.warning(
                    'Shape mismatch in pooled stack: expected %s, got %s for %s — skip',
                    total_count.shape, c.shape, p.name,
                )
                continue
            total_count += c
            mean_acc    += m * c

        if total_count is None:
            return None, None, None

        safe_count = np.where(total_count > 0, total_count, 1.0)
        pooled_mean = mean_acc / safe_count

        # Pass 2: compute pooled variance (same outlier mask as pass 1)
        var_acc = np.zeros_like(pooled_mean)
        for p in drp_paths:
            try:
                with rasterio.open(p) as src:
                    m = src.read(1).astype(np.float32)
                    s = src.read(2).astype(np.float32)
                    c = src.read(3).astype(np.float32)
            except Exception:
                continue
            if c.shape != pooled_mean.shape:
                continue
            m = self._clip_outliers(m)
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
        """Glob all PMP TIFs for a given (date, product)."""
        pmp_dir = (
            self.l3 / 'PMP'
            / f'{date.year:04d}' / f'{date.month:02d}' / f'{date.day:02d}'
        )
        if not pmp_dir.exists():
            return []
        return sorted(pmp_dir.rglob(f'*_{product}.tif'))

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
    def _read_meta(path: Path, band: int = 1) -> dict:
        """Read rasterio profile from a TIF, reset to single-band float32."""
        with rasterio.open(path) as src:
            meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': 'float32', 'nodata': float('nan')})
        return meta
