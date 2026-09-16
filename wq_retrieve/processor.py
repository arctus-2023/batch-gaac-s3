"""SceneProcessor — runs WQ algorithms on a single L2 scene, writes PMP TIFs."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

from .config import WQConfig, resolve_product
from .registry import get_algorithm
from .scene import MissingBandsError, WQScene
from .sensors import SensorSpec
from .io import write_wq_tif
from .algorithms.base import WQAlgorithm

logger = logging.getLogger(__name__)


class SceneProcessor:
    """Process a single *_GAAC scene directory → write PMP TIFs per product.

    Algorithms are chosen per sensor: a product may name one algorithm for OLCI
    and another for MSI or OLI (see `config.resolve_product`), and any algorithm
    whose family or band requirements the scene cannot satisfy is skipped with a
    warning rather than failing the scene.

    PMP storage path
    ----------------
    <l3_dir>/PMP/<YYYY>/<MM>/<DD>/<scene_stem>/<scene_stem>_<product>.tif
    """

    def __init__(self, cfg: WQConfig) -> None:
        self.cfg = cfg
        # {sensor key: {product: algorithm instance}} — built lazily per sensor
        self._algorithms: dict[str, dict[str, WQAlgorithm]] = {}
        from . import algorithms as _alg_pkg  # noqa: F401 — triggers registration

    def _algorithms_for(self, sensor: SensorSpec) -> dict[str, WQAlgorithm]:
        """Instantiate (and cache) the algorithm set this sensor should run."""
        cached = self._algorithms.get(sensor.key)
        if cached is not None:
            return cached

        built: dict[str, WQAlgorithm] = {}
        for prod, prod_cfg in self.cfg.wq_products.items():
            variant = resolve_product(prod_cfg, sensor)
            if variant is None:
                logger.debug('Product %r disabled for %s', prod, sensor.key)
                continue

            algo_name = variant['algorithm']
            try:
                AlgoCls = get_algorithm(prod, algo_name)
            except KeyError as exc:
                logger.error('Config error for product %r: %s', prod, exc)
                continue

            if not AlgoCls.supports(sensor):
                logger.warning(
                    'Product %r: algorithm %r is not calibrated for %s (families=%s) '
                    '— skip; set wq_products.%s.sensors.%s in the config',
                    prod, algo_name, sensor.key, AlgoCls.families, prod, sensor.family,
                )
                continue

            built[prod] = AlgoCls(**(variant.get('params') or {}))

        self._algorithms[sensor.key] = built
        return built

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
            scene = WQScene(
                rhow_path, mask_path,
                gaac_gen_dir=self.cfg.gaac_gen_dir,
                band_tolerance=self.cfg.band_tolerance,
                exclude_bits=self.cfg.mask_exclude_bits,
            )
            n_water = int(scene.water_mask.sum())
        except Exception as exc:
            logger.error('Failed to load scene %s: %s', scene_dir.name, exc)
            return {}

        if n_water == 0:
            logger.warning('Scene %s has 0 clear-water pixels — skip', scene.stem)
            return {}

        sensor = scene.sensor
        logger.info('Scene %s  sensor=%s  date=%s  kept_px=%d',
                    scene.stem, sensor.key, scene.date, n_water)
        logger.info('  mask: %s  exclude_bits=%s%s', scene.mask_source,
                    self.cfg.mask_exclude_bits,
                    f' (= {scene.mask_bits})' if scene.mask_bits is not None else '')

        algorithms = self._algorithms_for(sensor)
        if not algorithms:
            logger.warning('No applicable algorithms for %s (%s) — skip',
                           scene.stem, sensor.key)
            return {}

        outputs: dict[str, Path] = {}
        for prod, algo in algorithms.items():
            out_path = self._pmp_path(scene, prod)
            if out_path.exists() and not self.cfg.replace_output:
                logger.debug('PMP exists, skip: %s', out_path.name)
                outputs[prod] = out_path
                continue

            wavelengths = algo.bands_for(sensor.family)
            if not wavelengths:
                logger.warning('Product %r/%r declares no bands for family %s — skip',
                               prod, algo.name, sensor.family)
                continue

            try:
                bands = scene.select(wavelengths, quantity=algo.input_quantity)
            except MissingBandsError as exc:
                logger.warning('Product %r/%r on %s: %s — skip',
                               prod, algo.name, sensor.key, exc)
                continue

            mapping, _ = scene.resolve(wavelengths)
            substitutions = {nom: act for nom, act in mapping.items() if nom != act}
            if substitutions:
                logger.info('  %-12s band substitutions (nominal→actual): %s',
                            prod, substitutions)

            algo.bind(sensor)
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
                extra_tags={
                    'scene':  scene.stem,
                    'date':   str(scene.date),
                    'sensor': sensor.key,
                    'bands':  ','.join(str(mapping[wl]) for wl in wavelengths),
                    'mask_source': scene.mask_source,
                    'mask_exclude_bits': ('' if scene.mask_bits is None
                                          else str(scene.mask_bits)),
                },
            )
            n_valid = int(np.isfinite(result).sum())
            logger.info('  %-12s → %s  (algo=%s, valid_px=%d)',
                        prod, out_path.name, algo.name, n_valid)
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
        # A legacy mask file is only a fallback: current GAAC output carries the
        # register inside the rhow and deletes these files, so their absence is
        # normal. WQScene raises if the rhow has no flag band AND none exists.
        for candidate in (f'{base}_mask.tif', f'{base}_watermask.tif'):
            if (scene_dir / candidate).exists():
                return rhow_path, scene_dir / candidate
        return rhow_path, None

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
