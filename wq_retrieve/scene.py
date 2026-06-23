"""WQScene — lazy-loading adapter for a single L2 rhow + mask pair.

The Sentinel-3 OLCI L2 rhow file stores ρw (water-leaving reflectance, π·Rrs).
WQScene exposes both:
  .rrs  — Rrs = rhow / π  (for ratio / Rrs-calibrated algorithms)
  .rhow — raw ρw           (for Nechad/Dogliotti ρw-calibrated algorithms)

Both dicts have the same NaN mask applied: non-clear-water pixels are NaN.
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

# regex to parse band descriptions, e.g. 'rhow(443)' → 443
_DESC_RE = re.compile(r'rhow\((\d+)\)')

_PI = float(np.pi)

# rhow nodata value as written by gaac pipeline (from ac/output.py)
_RHOW_NODATA = 1.0


class WQScene:
    """Adapter that presents a single L2 scene as algorithm-ready arrays.

    Parameters
    ----------
    rhow_path : path to *_rhor_rhow.tif (15-band float32 ρw)
    mask_path : path to *_mask.tif or *_watermask.tif; if None, auto-detected
    gaac_gen_dir : optional path injected into sys.path for Cinputmask support
    """

    def __init__(
        self,
        rhow_path: str | Path,
        mask_path: str | Path | None = None,
        gaac_gen_dir: str | None = None,
    ) -> None:
        self.rhow_path = Path(rhow_path)
        self._mask_path = self._resolve_mask(mask_path)
        self._gaac_gen_dir = gaac_gen_dir
        self._rrs: dict[int, np.ndarray] | None = None
        self._rhow_dict: dict[int, np.ndarray] | None = None
        self._water_mask: np.ndarray | None = None
        self._meta: dict | None = None
        self.date: datetime.date = self._parse_date()

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def rrs(self) -> dict[int, np.ndarray]:
        """Rrs = ρw / π, keyed by wavelength (nm int). NaN on non-clear-water."""
        if self._rrs is None:
            self._load()
        return self._rrs  # type: ignore[return-value]

    @property
    def rhow(self) -> dict[int, np.ndarray]:
        """ρw (raw water-leaving reflectance), keyed by wavelength. NaN on non-clear-water."""
        if self._rhow_dict is None:
            self._load()
        return self._rhow_dict  # type: ignore[return-value]

    @property
    def water_mask(self) -> np.ndarray:
        """bool (H, W) — True where pixel is clear water."""
        if self._water_mask is None:
            self._load()
        return self._water_mask  # type: ignore[return-value]

    @property
    def meta(self) -> dict:
        """rasterio profile suitable for writing single-band outputs."""
        if self._meta is None:
            self._load()
        return self._meta  # type: ignore[return-value]

    @property
    def stem(self) -> str:
        """Scene name without _GAAC suffix, e.g. 'S3A_L1TOA_20250615T160535_...'."""
        name = self.rhow_path.parent.name
        return name.removesuffix('_GAAC') if name.endswith('_GAAC') else name

    # ── private helpers ───────────────────────────────────────────────────────

    def _parse_date(self) -> datetime.date:
        """Extract date from filename token at index 2.

        e.g. S3A_L1TOA_20250615T160535_997148+0000_JamesBay_300m_rhor_rhow.tif
             → tokens[2] = '20250615T160535' → 2025-06-15
        """
        tokens = self.rhow_path.name.split('_')
        try:
            tok = tokens[2]  # '20250615T160535'
            return datetime.date(int(tok[:4]), int(tok[4:6]), int(tok[6:8]))
        except (IndexError, ValueError):
            return datetime.date(1970, 1, 1)

    def _resolve_mask(self, mask_path: str | Path | None) -> Path:
        """Find the best available mask file in the same directory as rhow."""
        if mask_path is not None:
            return Path(mask_path)
        scene_dir = Path(self.rhow_path).parent
        # Reconstruct base name: strip _rhor_rhow.tif suffix
        base = self.rhow_path.name.replace('_rhor_rhow.tif', '').replace('_rhow.tif', '')
        preferred = scene_dir / f'{base}_mask.tif'
        fallback  = scene_dir / f'{base}_watermask.tif'
        if preferred.exists():
            return preferred
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f'No mask file found for scene {base!r} in {scene_dir}'
        )

    def _load(self) -> None:
        """Load rhow bands and water mask, apply mask, cache all results."""
        rhow_dict, meta = self._read_rhow()
        water_mask = self._read_mask()

        # Apply mask: NaN on non-clear-water pixels for every band
        invalid = ~water_mask
        for arr in rhow_dict.values():
            arr[invalid] = np.nan

        rrs_dict = {wl: (arr / _PI) for wl, arr in rhow_dict.items()}

        self._rhow_dict = rhow_dict
        self._rrs = rrs_dict
        self._water_mask = water_mask
        self._meta = meta

    def _read_rhow(self) -> tuple[dict[int, np.ndarray], dict]:
        """Read all bands from *_rhor_rhow.tif.

        Returns
        -------
        (rhow_by_wavelength, rasterio_meta)
        nodata pixels (value >= _RHOW_NODATA) are set to NaN.
        """
        rhow_dict: dict[int, np.ndarray] = {}
        with rasterio.open(self.rhow_path) as src:
            meta = src.meta.copy()
            for i in range(1, src.count + 1):
                desc = src.descriptions[i - 1] or ''
                m = _DESC_RE.match(desc)
                if not m:
                    continue
                wl = int(m.group(1))
                arr = src.read(i).astype(np.float32)
                arr[arr >= _RHOW_NODATA] = np.nan
                rhow_dict[wl] = arr

        meta.update({'count': 1, 'dtype': 'float32', 'nodata': float('nan')})
        return rhow_dict, meta

    def _read_mask(self) -> np.ndarray:
        """Return bool (H, W) — True = clear water pixel.

        Tries Cinputmask from gaac_gen (handles all schemes); falls back to
        manual tag inspection when gaac_gen is not on sys.path.
        """
        if self._gaac_gen_dir and self._gaac_gen_dir not in sys.path:
            sys.path.insert(0, self._gaac_gen_dir)

        try:
            from gaac.ac.inputs import Cinputmask  # type: ignore[import]
            cin = Cinputmask(str(self._mask_path))
            mask = cin.get_clearwater_mask()
            # Cinputmask may return (H, W) or (1, H, W)
            if mask.ndim == 3:
                mask = mask[0]
            return mask.astype(bool)
        except Exception:
            pass

        try:
            return self._read_mask_fallback()
        except Exception as exc:
            raise RuntimeError(
                f'Cannot read mask {self._mask_path}: {exc}'
            ) from exc

    def _read_mask_fallback(self) -> np.ndarray:
        """Manually replicate Cinputmask.get_clearwater_mask() logic.

        Supports three mask encodings produced by the GAAC pipeline:
          S3 scheme    : clear_water_value tag = 0 (nodata=255)
          SIMPLE scheme: clear_water_value tag = 100
          Legacy wm    : mask_items tag contains 'water_5' (0=null, 5=water)
        """
        with rasterio.open(self._mask_path) as src:
            data = src.read(1)
            tags = src.tags()

        if 'clear_water_value' in tags:
            cw_val = int(tags['clear_water_value'])
            return (data == cw_val).astype(bool)

        if 'mask_items' in tags:
            for item in tags['mask_items'].split(','):
                parts = item.rsplit('_', 1)
                if len(parts) == 2 and parts[0] == 'water':
                    return (data == int(parts[1])).astype(bool)

        # Absolute fallback: legacy watermask (0=null, 5=water)
        return (data == 5).astype(bool)
