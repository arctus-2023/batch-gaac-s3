#!/usr/bin/env python3
"""
batch_plot_s3_indices.py

Batch plot RGB, ENDSIII, and NDSI for Sentinel-3 OLCI L1 TOA GeoTIFFs.

Usage
-----
    python batch_plot_s3_indices.py <l1_dir> <output_dir> [options]

    l1_dir      directory containing L1 TOA *.tif files
    output_dir  directory to write PNG plots (created if absent)

Options
-------
    --dpi N       output resolution (default 150)

Indices
-------
    ENDSIII = (Oa12 - Oa16 + Oa20 - Oa21) / (Oa12 + Oa16 + Oa20 + Oa21)
              bands 754, 779, 939, 1016 nm
              contour at threshold = -0.01 (fixed) and 0

    NDSI    = (Oa17 - Oa21) / (Oa17 + Oa21)
              bands 865, 1016 nm
              contour at threshold = 0.03 (NDSI_OTSU fallback)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio


# ── band reading ──────────────────────────────────────────────────────────────

def _band_lookup(src):
    return {(d or '').lower(): i + 1 for i, d in enumerate(src.descriptions)}


def read_band(src, name, lookup=None):
    if lookup is None:
        lookup = _band_lookup(src)
    key = name.lower()
    if key not in lookup:
        raise ValueError(f"Band '{name}' not found. Available: {sorted(lookup)}")
    arr = src.read(lookup[key]).astype(np.float32)
    # replace fill values / zeros that indicate no-data
    arr[arr <= 0] = np.nan
    return arr


# ── indices ───────────────────────────────────────────────────────────────────

def compute_endsiii(src):
    lk = _band_lookup(src)
    r754  = read_band(src, 'rhot_754',  lk)
    r779  = read_band(src, 'rhot_779',  lk)
    r939  = read_band(src, 'rhot_939',  lk)
    r1016 = read_band(src, 'rhot_1016', lk)
    denom = r754 + r779 + r939 + r1016
    with np.errstate(invalid='ignore', divide='ignore'):
        idx = (r754 - r779 + r939 - r1016) / denom
    idx[~np.isfinite(idx)] = np.nan
    return idx


def compute_ndsi(src):
    lk = _band_lookup(src)
    r865  = read_band(src, 'rhot_865',  lk)   # Oa17
    r1016 = read_band(src, 'rhot_1016', lk)   # Oa21
    denom = r865 + r1016
    with np.errstate(invalid='ignore', divide='ignore'):
        idx = (r865 - r1016) / denom
    idx[~np.isfinite(idx)] = np.nan
    return idx


def compute_rgb(src):
    lk = _band_lookup(src)
    r = read_band(src, 'rhot_665', lk)   # Oa08
    g = read_band(src, 'rhot_560', lk)   # Oa06
    b = read_band(src, 'rhot_490', lk)   # Oa04
    rgb = np.stack([_pstretch(r), _pstretch(g), _pstretch(b)], axis=-1)
    return rgb


def _index_range(arr, lo=2, hi=98):
    '''Return (vmin, vmax) from percentile stretch of finite values.'''
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


def _pstretch(arr, lo=2, hi=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    vmin, vmax = np.nanpercentile(valid, [lo, hi])
    if vmax <= vmin:
        return np.zeros_like(arr)
    out = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    out[~np.isfinite(arr)] = 0
    return out


# ── GeoTIFF output ────────────────────────────────────────────────────────────

def save_index_tif(arr, src_profile, out_path, description):
    profile = src_profile.copy()
    profile.update(driver='GTiff', count=1, dtype='float32', nodata=np.nan)
    # drop any per-band metadata keys not valid for single-band output
    profile.pop('compress', None)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.set_band_description(1, description)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_scene(tif_path, out_png, out_endsiii_tif, out_ndsi_tif, args):
    with rasterio.open(tif_path) as src:
        rgb     = compute_rgb(src)
        endsiii = compute_endsiii(src)
        ndsi    = compute_ndsi(src)
        profile = src.profile.copy()

    save_index_tif(endsiii, profile, out_endsiii_tif,
                   'ENDSIII (Oa12-Oa16+Oa20-Oa21)/(Oa12+Oa16+Oa20+Oa21)')
    save_index_tif(ndsi, profile, out_ndsi_tif,
                   'NDSI (Oa17-Oa21)/(Oa17+Oa21)')

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(tif_path.stem, fontsize=9, y=1.005)

    # ── RGB ───────────────────────────────────────────────────────────────────
    axes[0].imshow(rgb, interpolation='nearest')
    axes[0].set_title('RGB  (665 / 560 / 490 nm, 2–98% stretch)', fontsize=8)
    axes[0].axis('off')

    # ── ENDSIII ───────────────────────────────────────────────────────────────
    e_vmin, e_vmax = _index_range(endsiii)
    im1 = axes[1].imshow(
        endsiii, cmap='gray',
        vmin=e_vmin, vmax=e_vmax,
        interpolation='nearest',
    )
    axes[1].set_title(
        f'ENDSIII  (Oa12−Oa16+Oa20−Oa21)/(Oa12+Oa16+Oa20+Oa21)'
        f'\n[{e_vmin:.3f} … {e_vmax:.3f}]', fontsize=8
    )
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)

    # ── NDSI ──────────────────────────────────────────────────────────────────
    n_vmin, n_vmax = _index_range(ndsi)
    im2 = axes[2].imshow(
        ndsi, cmap='gray',
        vmin=n_vmin, vmax=n_vmax,
        interpolation='nearest',
    )
    axes[2].set_title(
        f'NDSI  (Oa17−Oa21)/(Oa17+Oa21)  [865 / 1016 nm]'
        f'\n[{n_vmin:.3f} … {n_vmax:.3f}]', fontsize=8
    )
    axes[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.035, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Batch plot RGB, ENDSIII and NDSI for S3 OLCI L1 TOA GeoTIFFs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('l1_dir',    help='Directory containing L1 TOA *.tif files')
    ap.add_argument('output_dir', help='Directory to write PNG plots')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    l1_dir  = Path(args.l1_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(l1_dir.glob('*.tif'))
    if not scenes:
        print(f'No *.tif files found in {l1_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(scenes)} scene(s) → {out_dir}')
    ok = fail = 0
    for tif in scenes:
        stem        = tif.stem
        out_png     = out_dir / f'{stem}_indices.png'
        out_endsiii = out_dir / f'{stem}_ENDSIII.tif'
        out_ndsi    = out_dir / f'{stem}_NDSI.tif'
        print(f'  {tif.name} ...', end=' ', flush=True)
        try:
            plot_scene(tif, out_png, out_endsiii, out_ndsi, args)
            print('done')
            ok += 1
        except Exception as exc:
            print(f'FAILED: {exc}')
            fail += 1

    print(f'\nDone — {ok} plotted, {fail} failed.')


if __name__ == '__main__':
    main()
