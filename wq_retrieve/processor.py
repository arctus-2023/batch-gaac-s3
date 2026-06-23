"""SceneProcessor — runs WQ algorithms on a single L2 scene, writes PMP TIFs."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

from .config import WQConfig
from .registry import get_algorithm
from .scene import WQScene
from .io import write_wq_tif

logger = logging.getLogger(__name__)


class SceneProcessor:
    """Process a single *_GAAC scene directory → write PMP TIFs per product.

    PMP storage path
    ----------------
    <l3_dir>/PMP/<YYYY>/<MM>/<DD>/<scene_stem>/<scene_stem>_<product>.tif
    """

    def __init__(self, cfg: WQConfig) -> None:
        self.cfg = cfg
        self._algorithms: dict = {}   # {product: (algo_instance, algo_name)}
        self._build_algorithms()

    def _build_algorithms(self) -> None:
        from . import algorithms as _alg_pkg  # noqa: F401 — triggers registration
        for prod, prod_cfg in self.cfg.wq_products.items():
            if not prod_cfg.get('enabled', True):
                continue
            algo_name = prod_cfg['algorithm']
            params = prod_cfg.get('params') or {}
            try:
                AlgoCls = get_algorithm(prod, algo_name)
            except KeyError as exc:
                logger.error('Config error for product %r: %s', prod, exc)
                continue
            self._algorithms[prod] = AlgoCls(**params)

    def process_scene(self, scene_dir: str | Path) -> dict[str, Path]:
        """Run all enabled algorithms on the scene; write PMP TIFs.

        Parameters
        ----------
        scene_dir : path to a *_GAAC/ directory containing *_rhor_rhow.tif

        Returns
        -------
        dict {product_name: output_path} for each successfully written product
        """
        scene_dir = Path(scene_dir)
        rhow_path, mask_path = self._find_inputs(scene_dir)
        if rhow_path is None:
            logger.warning('No *_rhor_rhow.tif found in %s — skip', scene_dir)
            return {}

        try:
            scene = WQScene(rhow_path, mask_path, gaac_gen_dir=self.cfg.gaac_gen_dir)
            n_water = int(scene.water_mask.sum())
        except Exception as exc:
            logger.error('Failed to load scene %s: %s', scene_dir.name, exc)
            return {}

        if n_water == 0:
            logger.warning('Scene %s has 0 clear-water pixels — skip', scene.stem)
            return {}

        logger.info('Scene %s  date=%s  water_px=%d', scene.stem, scene.date, n_water)

        outputs: dict[str, Path] = {}
        for prod, algo in self._algorithms.items():
            out_path = self._pmp_path(scene, prod)
            if out_path.exists() and not self.cfg.replace_output:
                logger.debug('PMP exists, skip: %s', out_path.name)
                outputs[prod] = out_path
                continue

            # Check required bands
            available = set(scene.rrs.keys())  # same keys in both rrs and rhow
            missing = [b for b in algo.required_bands if b not in available]
            if missing:
                logger.warning(
                    'Product %r/%r: required bands %s not in scene (available: %s) — skip',
                    prod, algo.name, missing, sorted(available)
                )
                continue

            # Select band dict by algorithm's declared input_quantity
            if algo.input_quantity == 'rhow':
                bands = {wl: scene.rhow[wl] for wl in algo.required_bands}
            else:
                bands = {wl: scene.rrs[wl] for wl in algo.required_bands}

            try:
                result = algo.compute(bands)
            except Exception as exc:
                logger.error('Algorithm %r/%r failed on %s: %s',
                             prod, algo.name, scene.stem, exc)
                continue

            write_wq_tif(
                path=out_path,
                data=result,
                meta=scene.meta,
                product=prod,
                algorithm=algo.name,
                units=algo.units,
                extra_tags={'scene': scene.stem, 'date': str(scene.date)},
            )
            n_valid = int(np.isfinite(result).sum())
            logger.info('  %-12s → %s  (valid_px=%d)', prod, out_path.name, n_valid)
            outputs[prod] = out_path

        return outputs

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_inputs(
        self, scene_dir: Path
    ) -> tuple[Path | None, Path | None]:
        """Locate *_rhor_rhow.tif and best available mask file in scene_dir."""
        rhow_files = sorted(scene_dir.glob('*_rhor_rhow.tif'))
        if not rhow_files:
            return None, None
        rhow_path = rhow_files[0]
        base = rhow_path.name.replace('_rhor_rhow.tif', '')
        mask_path = scene_dir / f'{base}_mask.tif'
        if not mask_path.exists():
            mask_path = scene_dir / f'{base}_watermask.tif'
        if not mask_path.exists():
            logger.warning('No mask file found for %s', base)
            mask_path = None
        return rhow_path, mask_path

    def _pmp_path(self, scene: WQScene, product: str) -> Path:
        """Build PMP output path: <l3>/PMP/YYYY/MM/DD/<scene_stem>/<stem>_<product>.tif"""
        d = scene.date
        return (
            Path(self.cfg.l3_dir)
            / 'PMP'
            / f'{d.year:04d}'
            / f'{d.month:02d}'
            / f'{d.day:02d}'
            / scene.stem
            / f'{scene.stem}_{product}.tif'
        )
