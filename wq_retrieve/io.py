"""Rasterio I/O helpers for writing WQ product GeoTIFFs.

Follows the same compression / tagging conventions as gaac_gen/src/gaac/ac/output.py:
  compress=lzw, tiled, blockxsize=512, blockysize=512, BIGTIFF=YES
"""

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import rasterio

from . import __version__

_WRITE_OPTIONS = {
    'compress':   'lzw',
    'tiled':       True,
    'blockxsize':  512,
    'blockysize':  512,
    'BIGTIFF':    'YES',
}


def write_wq_tif(
    path: str | Path,
    data: np.ndarray,
    meta: dict,
    product: str,
    algorithm: str,
    units: str,
    extra_tags: dict | None = None,
) -> None:
    """Write a single-band float32 WQ product GeoTIFF.

    Parameters
    ----------
    path     : output file path (parent dirs created automatically)
    data     : float32 array (H, W); NaN where invalid
    meta     : rasterio profile from WQScene.meta (provides crs, transform, H, W)
    product  : product name, e.g. 'chla'
    algorithm: algorithm key, e.g. 'gons2005'
    units    : physical units string, e.g. 'mg m-3'
    extra_tags: optional dict of additional tags to write
    """
    out_meta = meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count':  1,
        'dtype':  'float32',
        'nodata': float('nan'),
        **_WRITE_OPTIONS,
    })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, 'w', **out_meta) as dst:
        dst.write(data.astype(np.float32), 1)
        dst.set_band_description(1, f'{product}({units})')
        dst.update_tags(
            product=product,
            algorithm=algorithm,
            units=units,
        )
        dst.update_tags(
            ns='software',
            software_name='wq_retrieve',
            version=__version__,
            author='Arctus Inc.',
            description='ARCTUS water quality retrieval chain',
        )
        if extra_tags:
            dst.update_tags(**extra_tags)


def write_drp_tif(
    path: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
    count: np.ndarray,
    meta: dict,
    product: str,
    algorithm: str,
    units: str,
    period: str,
    date_label: str,
) -> None:
    """Write a 3-band DRP composite GeoTIFF.

    Bands
    -----
    1: mean  — temporal mean of valid observations
    2: std   — temporal standard deviation (population)
    3: count — number of valid scene observations contributing

    Parameters
    ----------
    period     : 'daily' | 'monthly' | 'yearly'
    date_label : e.g. '20250710', '202507', '2025'
    """
    out_meta = meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count':  3,
        'dtype':  'float32',
        'nodata': float('nan'),
        **_WRITE_OPTIONS,
    })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, 'w', **out_meta) as dst:
        dst.write(mean.astype(np.float32),  1)
        dst.write(std.astype(np.float32),   2)
        dst.write(count.astype(np.float32), 3)
        dst.set_band_description(1, f'{product}_mean({units})')
        dst.set_band_description(2, f'{product}_std({units})')
        dst.set_band_description(3, f'{product}_count')
        dst.update_tags(
            product=product,
            algorithm=algorithm,
            units=units,
            period=period,
            date_label=date_label,
        )
        dst.update_tags(
            ns='software',
            software_name='wq_retrieve',
            version=__version__,
        )
