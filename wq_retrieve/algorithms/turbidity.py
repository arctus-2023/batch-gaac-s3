"""Turbidity retrieval algorithms [FNU].

Algorithms
----------
dogliotti2015_t  : switching red/NIR turbidity — Dogliotti et al. 2015
                   gold-standard single-algorithm for all coastal/estuarine waters
nechad2016_olci  : OLCI-specific band LUT — Nechad et al. 2016
                   uses Oa08/Oa11/Oa17 three-band switching

NOTE: Both algorithms use input_quantity='rhow' (ρw = π·Rrs).
"""

from __future__ import annotations
import numpy as np
from ..registry import register_algorithm
from .base import WQAlgorithm
from .spm import _DOGLIOTTI_RED, _DOGLIOTTI_NIR, _dogliotti_t


@register_algorithm('turbidity', 'dogliotti2015_t')
class Dogliotti2015Turbidity(WQAlgorithm):
    """Switching red/NIR turbidity algorithm — gold standard for coastal waters.

    T [FNU] = (AT · ρw(λ)) / (1 − ρw(λ)/CT)

    Low turbidity (< t_low FNU):  uses λ=665 nm (Oa08, proxy for original 645 nm)
    High turbidity (> t_high FNU): uses λ=865 nm (Oa17)
    Blend zone: linear interpolation

    Original calibration used 645 nm; OLCI lacks this band. Oa08 (665 nm) is
    used with the published coefficients. Recalibrate AT_red if local in situ
    FNU data are available for arctic mineral-particle waters.
    """
    product = 'turbidity'
    name = 'dogliotti2015_t'
    units = 'FNU'
    reference = ('Dogliotti et al. (2015). Remote Sensing of Environment 156:157–168. '
                 'doi:10.1016/j.rse.2014.09.020')
    required_bands = [665, 865]
    input_quantity = 'rhow'

    _DEFAULTS = {
        'AT_red': _DOGLIOTTI_RED['AT'],
        'CT_red': _DOGLIOTTI_RED['CT'],
        'AT_nir': _DOGLIOTTI_NIR['AT'],
        'CT_nir': _DOGLIOTTI_NIR['CT'],
        't_low':  7.0,
        't_high': 20.0,
    }

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        rho665, rho865 = bands[665], bands[865]
        result = _dogliotti_t(rho665, rho865,
                              self.AT_red, self.CT_red,
                              self.AT_nir, self.CT_nir,
                              self.t_low, self.t_high)
        return np.where(result > 0, result, np.nan).astype(np.float32)


@register_algorithm('turbidity', 'nechad2016_olci')
class Nechad2016OLCITurbidity(WQAlgorithm):
    """Three-band switching turbidity using OLCI-specific Nechad 2016 LUT.

    T [FNU] = (A · ρw(λ)) / (1 − ρw(λ)/C)

    Three-band switching scheme:
      ρw(665)  < rho_665_max → Oa08 branch (AT=188.49, CT=0.164)
      ρw(709)  < rho_709_max → Oa11 branch (AT=261.7,  CT=0.212)
      otherwise              → Oa17 branch (AT=3265,   CT=0.212)

    Oa11 (708.75 nm) is a unique OLCI advantage over MODIS/Landsat; it fills
    the transitional 5–50 FNU range especially useful for subarctic river plumes.

    Coefficients from Nechad et al. (2016) CoastColour Round Robin LUT.
    """
    product = 'turbidity'
    name = 'nechad2016_olci'
    units = 'FNU'
    reference = ('Nechad et al. (2016). Earth System Science Data 8:173–196. '
                 'doi:10.5194/essd-8-173-2016')
    required_bands = [665, 709, 865]
    input_quantity = 'rhow'

    _DEFAULTS = {
        'AT_665':      188.49,
        'CT_665':      0.164,
        'AT_709':      261.7,
        'CT_709':      0.212,
        'AT_865':      3265.0,
        'CT_865':      0.212,
        'rho_665_max': 0.05,   # ρw(665) threshold to switch from Oa08 to Oa11
        'rho_709_max': 0.12,   # ρw(709) threshold to switch from Oa11 to Oa17
    }

    def _branch(self, rhow: np.ndarray, AT: float, CT: float) -> np.ndarray:
        denom = 1.0 - rhow / CT
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(denom > 0, (AT * rhow) / denom, np.nan)

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        rho665, rho709, rho865 = bands[665], bands[709], bands[865]
        T665 = self._branch(rho665, self.AT_665, self.CT_665)
        T709 = self._branch(rho709, self.AT_709, self.CT_709)
        T865 = self._branch(rho865, self.AT_865, self.CT_865)

        result = np.where(
            rho665 < self.rho_665_max, T665,
            np.where(rho709 < self.rho_709_max, T709, T865)
        )
        return np.where(result > 0, result, np.nan).astype(np.float32)
