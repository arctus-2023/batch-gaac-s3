"""WQScene — lazy-loading adapter for a single L2 rhow + mask pair.

The GAAC L2 rhow file stores ρw (water-leaving reflectance, π·Rrs) for every
supported instrument — Sentinel-3 OLCI, Sentinel-2 MSI, and Landsat-8/9
OLI/OLI-2.  WQScene exposes both:
  .rrs  — Rrs = rhow / π  (for ratio / Rrs-calibrated algorithms)
  .rhow — raw ρw           (for Nechad/Dogliotti ρw-calibrated algorithms)

Both dicts have the same NaN mask applied: excluded pixels are NaN, and both
are keyed by the band centres the file actually carries (e.g. 655 nm on OLI,
665 nm on OLCI).  Use `select()` to fetch algorithm-nominal wavelengths; it
resolves them onto the nearest available band.

Masking
-------
GAAC writes its uint16 bit register into the rhow file itself, as a band named by
the `flag_band` tag, and then deletes *_mask.tif and *_watermask.tif.  That band
is therefore the mask source: a pixel is excluded when ANY of the chosen bits is
set.  Which bits is the caller's choice (`exclude_bits`), given as bit names --
resolved through the file's own `flag_bits` tag -- or bit numbers, plus the
keyword `opt_exclude` for the register's own `opt_exclude_bits` (GAAC's notion of
clear water).  The choice matters for water quality: `nonwater_swir` fires on
bright-NIR turbid water, precisely the pixels an SPM or turbidity product wants.

An L2 scene written before the flag band existed has no such band; for those the
old *_mask.tif / *_watermask.tif path is still used.
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

from .sensors import DEFAULT_TOLERANCE, SensorSpec, detect_sensor, resolve_wavelengths

# regex to parse band descriptions, e.g. 'rhow(443)' → 443
_DESC_RE = re.compile(r'rhow\((\d+)\)')

# YYYYMMDD anywhere in the filename, e.g. '20250615T160535' or '_20250628_'
_DATE_RE = re.compile(r'(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)')

_PI = float(np.pi)

# rhow nodata value as written by gaac pipeline (from ac/output.py)
_RHOW_NODATA = 1.0


def parse_scene_date(name: str) -> datetime.date | None:
    """First valid YYYYMMDD token in a scene or file name; None if there is none.

    Handles every GAAC naming convention in use:
        S3A_L1TOA_20250615T160535_997148+0000_JamesBay_300m_rhor_rhow.tif
        S2_L1TOA_20170831T1628_La_Grande_S2_grid_60m_rhor_rhow.tif
        LC08_L1TOA_20250628_326451_30m_rhor_rhow.tif
    """
    for m in _DATE_RE.finditer(name):
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
    return None



def flag_bit_names(tags: dict) -> dict[str, int]:
    """name -> bit, from a GAAC file's `flag_bits` tag ('land:7,cloud:8,...')."""
    names: dict[str, int] = {}
    for item in (tags.get('flag_bits') or '').split(','):
        if ':' in item:
            name, bit = item.rsplit(':', 1)
            names[name.strip()] = int(bit)
    return names


def resolve_exclude_bits(spec, tags: dict) -> int:
    """Turn a user's bit selection into one integer mask.

    `spec` is a name, a bit number, the keyword 'opt_exclude', or a list of any
    of those.  Names are looked up in the file's own `flag_bits` tag, so a file
    keeps decoding by the layout it was written with.  An unknown name is an
    error rather than a silent no-op -- a typo in the config would otherwise
    quietly keep every pixel it was meant to drop.
    """
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    names = flag_bit_names(tags)
    value = 0
    for item in items:
        if isinstance(item, (int, np.integer)) or (isinstance(item, str) and item.isdigit()):
            bit = int(item)
            if not 0 <= bit < 16:
                raise ValueError(f'flag bit {bit} is outside the uint16 register (0-15)')
            value |= 1 << bit
        elif str(item).lower() == 'opt_exclude':
            if 'opt_exclude_bits' not in tags:
                raise ValueError("'opt_exclude' requested but the file has no "
                                 "opt_exclude_bits tag")
            value |= int(tags['opt_exclude_bits'])
        elif item in names:
            value |= 1 << names[item]
        else:
            raise ValueError(
                f'unknown flag bit {item!r}; this file defines '
                f'{", ".join(f"{n}={b}" for n, b in sorted(names.items(), key=lambda kv: kv[1]))}'
                f' (or use a bit number, or opt_exclude)')
    return value

class MissingBandsError(LookupError):
    """Raised when a scene has no band within tolerance of a requested wavelength."""

    def __init__(self, missing: list[int], available: list[int]) -> None:
        super().__init__(
            f'no band within tolerance of {missing} nm '
            f'(scene carries {available} nm)'
        )
        self.missing = missing
        self.available = available


class WQScene:
    """Adapter that presents a single L2 scene as algorithm-ready arrays.

    Parameters
    ----------
    rhow_path : path to *_rhor_rhow.tif (float32 ρw, one band per wavelength)
    mask_path : legacy *_mask.tif or *_watermask.tif, used only when the rhow file
        carries no flag band; if None, auto-detected when needed
    gaac_gen_dir : optional path injected into sys.path for Cinputmask support
    band_tolerance : max |nominal − actual| accepted when resolving bands [nm]
    exclude_bits : bits of the flag band that exclude a pixel -- names, numbers,
        and/or 'opt_exclude'; see the module docstring
    """

    def __init__(
        self,
        rhow_path: str | Path,
        mask_path: str | Path | None = None,
        gaac_gen_dir: str | None = None,
        band_tolerance: int = DEFAULT_TOLERANCE,
        exclude_bits: 'str | int | list[str | int]' = 'opt_exclude',
    ) -> None:
        self.rhow_path = Path(rhow_path)
        # Resolved lazily: a scene carrying a flag band has no mask file at all,
        # because GAAC deletes it, so looking for one up front would fail.
        self._mask_path_arg = mask_path
        self._mask_path: Path | None = None
        self.exclude_bits = exclude_bits
        #: what the mask was read from, and the bit value applied -- for logs/tags
        self.mask_source: str = ''
        self.mask_bits: int | None = None
        self._gaac_gen_dir = gaac_gen_dir
        self.band_tolerance = int(band_tolerance)
        self._rrs: dict[int, np.ndarray] | None = None
        self._rhow_dict: dict[int, np.ndarray] | None = None
        self._water_mask: np.ndarray | None = None
        self._meta: dict | None = None
        self._sensor: SensorSpec | None = None
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
    def sensor(self) -> SensorSpec:
        """Instrument that acquired this scene (from the file's `sensor` tag)."""
        if self._sensor is None:
            self._sensor = self._read_sensor()
        return self._sensor

    @property
    def wavelengths(self) -> list[int]:
        """Band centres (nm) the scene actually carries, ascending."""
        return sorted(self.rhow)

    @property
    def stem(self) -> str:
        """Scene name without _GAAC suffix, e.g. 'S3A_L1TOA_20250615T160535_...'."""
        name = self.rhow_path.parent.name
        return name.removesuffix('_GAAC') if name.endswith('_GAAC') else name

    # ── band access ───────────────────────────────────────────────────────────

    def resolve(self, wavelengths: list[int]) -> tuple[dict[int, int], list[int]]:
        """Map nominal wavelengths onto this scene's nearest actual bands.

        Returns ``({nominal: actual}, missing_nominals)``.
        """
        return resolve_wavelengths(self.wavelengths, wavelengths, self.band_tolerance)

    def select(
        self, wavelengths: list[int], quantity: str = 'Rrs'
    ) -> dict[int, np.ndarray]:
        """Return arrays for `wavelengths`, keyed by the *nominal* wavelength.

        `quantity` is 'Rrs' (default) or 'rhow'.  Raises MissingBandsError when
        any requested wavelength has no band within `band_tolerance`.
        """
        mapping, missing = self.resolve(wavelengths)
        if missing:
            raise MissingBandsError(missing, self.wavelengths)
        source = self.rhow if quantity == 'rhow' else self.rrs
        return {nominal: source[actual] for nominal, actual in mapping.items()}

    # ── private helpers ───────────────────────────────────────────────────────

    def _parse_date(self) -> datetime.date:
        """Acquisition date from the filename; epoch when the name carries none."""
        return parse_scene_date(self.rhow_path.name) or datetime.date(1970, 1, 1)

    def _read_sensor(self) -> SensorSpec:
        """Read the `sensor` tag from the rhow file; fall back to the scene name."""
        tag = None
        try:
            with rasterio.open(self.rhow_path) as src:
                tag = src.tags().get('sensor')
        except Exception:
            pass
        return detect_sensor(tag, self.rhow_path.name)

    def _resolve_mask(self, mask_path: str | Path | None) -> Path:
        """Find a legacy mask file next to the rhow (pre-flag-band L2 only)."""
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
            f'{self.rhow_path.name} has no flag band and no mask file '
            f'({base}_mask.tif / {base}_watermask.tif) was found in {scene_dir}'
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
            self._sensor = detect_sensor(src.tags().get('sensor'), self.rhow_path.name)
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

    # ── flag band ─────────────────────────────────────────────────────────────

    def _read_flag_band_mask(self) -> np.ndarray | None:
        """Keep-mask from the rhow flag band, or None when the file has none.

        A pixel is kept when none of `exclude_bits` is set. The band is float32 on
        disk (it shares the rhow dtype) but holds exact integers, so it is cast
        back to uint16 before the bitwise test.
        """
        with rasterio.open(self.rhow_path) as src:
            tags = src.tags()
            band = tags.get('flag_band')
            if not band:
                return None
            flags = src.read(int(band))

        # The flag band inherits the rhow nodata (1.0), so a masked read would
        # drop every pixel whose only set bit is nonwater_swir; read it raw.
        flags = np.nan_to_num(flags, nan=0.0).astype(np.uint16)
        bits = resolve_exclude_bits(self.exclude_bits, tags)
        self.mask_bits = bits
        self.mask_source = f'flag band {band} of {self.rhow_path.name}'
        return (flags & np.uint16(bits)) == 0

    def _read_mask(self) -> np.ndarray:
        """Return bool (H, W) — True = pixel is kept.

        The flag band in the rhow file is the source whenever it exists. Only an
        older L2 scene without one falls back to *_mask.tif / *_watermask.tif.
        """
        flagged = self._read_flag_band_mask()
        if flagged is not None:
            return flagged

        self._mask_path = self._resolve_mask(self._mask_path_arg)
        self.mask_source = self._mask_path.name
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
          SIMPLE scheme: clear_water_value tag = 100  (S2 / Landsat)
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
