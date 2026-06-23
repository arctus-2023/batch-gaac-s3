"""CDOM absorption coefficient retrieval algorithms (aCDOM at 443 nm, m⁻¹).

Algorithms
----------
glukhovets2020 : empirical log-linear band ratio (Rrs443/Rrs490)
                 Glukhovets et al. 2020, Remote Sensing, Arctic OLCI
mabit2022      : empirical power-law band ratio (Rrs443/Rrs560)
                 Mabit et al. 2022, Frontiers in Remote Sensing
                 calibrated on eastern James Bay + St. Lawrence estuary
"""

from __future__ import annotations
import numpy as np
from ..registry import register_algorithm
from .base import WQAlgorithm


@register_algorithm('cdom', 'glukhovets2020')
class Glukhovets2020CDOM(WQAlgorithm):
    """Semi-empirical log-linear CDOM algorithm for Arctic OLCI.

    log10(aCDOM(443)) = a0 + a1 * log10(Rrs(443) / Rrs(490))

    Default coefficients from Glukhovets et al. (2020) Table 3 (best-fit
    for 442/490 ratio across White Sea, Barents, Kara, and Laptev Seas).
    """
    product = 'cdom'
    name = 'glukhovets2020'
    units = 'm-1'
    reference = ('Glukhovets et al. (2020). Remote Sensing 12(19):3210. '
                 'doi:10.3390/rs12193210')
    required_bands = [443, 490]
    input_quantity = 'Rrs'

    _DEFAULTS = {'a0': -0.80, 'a1': -1.65}

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        r443, r490 = bands[443], bands[490]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(r490 > 0, r443 / r490, np.nan)
            log_ratio = np.where(ratio > 0, np.log10(ratio), np.nan)
            log_cdom = self.a0 + self.a1 * log_ratio
            result = np.power(10.0, log_cdom)
        return np.where(result > 0, result, np.nan).astype(np.float32)


@register_algorithm('cdom', 'mabit2022')
class Mabit2022CDOM(WQAlgorithm):
    """Power-law CDOM algorithm calibrated for Québec coastal / James Bay waters.

    aCDOM(443) = alpha * (Rrs(443) / Rrs(560)) ^ beta

    Default coefficients from Mabit et al. (2022) Table 3 (CDOM algorithm
    calibrated on eastern James Bay and lower St. Lawrence estuary field data).
    """
    product = 'cdom'
    name = 'mabit2022'
    units = 'm-1'
    reference = ('Mabit et al. (2022). Frontiers in Remote Sensing 3:834908. '
                 'doi:10.3389/frsen.2022.834908')
    required_bands = [443, 560]
    input_quantity = 'Rrs'

    _DEFAULTS = {'alpha': 0.85, 'beta': -1.40}

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        r443, r560 = bands[443], bands[560]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(r560 > 0, r443 / r560, np.nan)
            result = self.alpha * np.power(ratio, self.beta)
        return np.where(result > 0, result, np.nan).astype(np.float32)
