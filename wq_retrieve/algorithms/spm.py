"""Suspended Particulate Matter retrieval algorithms [g m⁻³].

Algorithms
----------
nechad2010   : generic single-band semi-empirical (ρw at 665 or 865 nm)
               Nechad et al. 2010, Remote Sensing of Environment
dogliotti2015: switching red/NIR algorithm (ρw at 665 + 865 nm)
               Dogliotti et al. 2015, Remote Sensing of Environment
doxaran2012  : power-law NIR/VIS ratio — calibrated on Mackenzie Arctic plume
               Doxaran et al. 2012, Biogeosciences
mabit_powerlaw: single-band log-power form used operationally for MSI and OLI
               (SAMBA / Eeyou-Sat L3 chain); reads the red-edge band on MSI and
               the red band on OLI, with a different coefficient pair for each

NOTE: nechad2010 and dogliotti2015 use input_quantity='rhow' (ρw = π·Rrs)
      because their calibration coefficients are formulated in terms of ρw.
      doxaran2012 uses a band ratio (Rrs865/Rrs560); π cancels → 'Rrs'.
"""

from __future__ import annotations
import numpy as np
from ..registry import register_algorithm
from ..sensors import FAMILY_MSI, FAMILY_OLI
from .base import WQAlgorithm

# Nechad2010 Table 2 band-specific coefficients (ρw formulation)
_NECHAD_665 = {'A': 355.85, 'C': 0.1728, 'B': 1.74}
_NECHAD_865 = {'A': 3077.2, 'C': 0.2112, 'B': -2.40}

# Dogliotti2015 Table 1 (ρw, calibrated at 645 nm → re-used here at 665 nm)
_DOGLIOTTI_RED = {'AT': 228.1,  'CT': 0.1641}   # 665 nm (proxy for 645 nm)
_DOGLIOTTI_NIR = {'AT': 3078.9, 'CT': 0.2112}   # 865 nm


def _nechad_formula(
    rhow: np.ndarray, A: float, C: float, B: float
) -> np.ndarray:
    """SPM or Turbidity from Nechad semi-empirical formula using ρw."""
    denom = 1.0 - rhow / C
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denom > 0, (A * rhow) / denom + B, np.nan)
    return result


def _dogliotti_t(rhow_red: np.ndarray, rhow_nir: np.ndarray,
                 AT_red: float, CT_red: float,
                 AT_nir: float, CT_nir: float,
                 t_low: float = 7.0, t_high: float = 20.0) -> np.ndarray:
    """Switching turbidity (FNU) from Dogliotti et al. (2015).

    Uses red band for low turbidity and NIR band for high turbidity,
    with a linear blend between t_low and t_high [FNU].
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        d_red = 1.0 - rhow_red / CT_red
        d_nir = 1.0 - rhow_nir / CT_nir
        T_red = np.where(d_red > 0, (AT_red * rhow_red) / d_red, np.nan)
        T_nir = np.where(d_nir > 0, (AT_nir * rhow_nir) / d_nir, np.nan)

    # linear blend fraction: 0 = pure red, 1 = pure NIR
    f = np.clip((T_red - t_low) / (t_high - t_low), 0.0, 1.0)

    T_blend = (1.0 - f) * T_red + f * T_nir

    # Force pure branches where we're clearly outside blend zone
    result = np.where(T_red <= t_low, T_red,
                      np.where(T_red >= t_high, T_nir, T_blend))
    return result


@register_algorithm('spm', 'nechad2010')
class Nechad2010SPM(WQAlgorithm):
    """Generic single-band semi-empirical SPM algorithm.

    SPM = (A · ρw(λ)) / (1 − ρw(λ)/C) + B

    Uses 665 nm band by default. Switches to 865 nm when ρw(665) exceeds
    `switch_threshold` (default 0.05, empirically ~50 g m⁻³).
    Recalibrate A and C coefficients with local in situ SPM data for best
    accuracy in James Bay / Hudson Bay mineral-particle waters.
    """
    product = 'spm'
    name = 'nechad2010'
    units = 'g m-3'
    reference = ('Nechad et al. (2010). Remote Sensing of Environment 114(4):854–866. '
                 'doi:10.1016/j.rse.2009.11.022')
    required_bands = [665, 865]
    input_quantity = 'rhow'

    _DEFAULTS = {
        'A_665':          _NECHAD_665['A'],
        'C_665':          _NECHAD_665['C'],
        'B_665':          _NECHAD_665['B'],
        'A_865':          _NECHAD_865['A'],
        'C_865':          _NECHAD_865['C'],
        'B_865':          _NECHAD_865['B'],
        'switch_threshold': 0.05,   # ρw(665) threshold to switch to 865 nm band
    }

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        rho665, rho865 = bands[665], bands[865]
        spm_665 = _nechad_formula(rho665, self.A_665, self.C_665, self.B_665)
        spm_865 = _nechad_formula(rho865, self.A_865, self.C_865, self.B_865)
        result = np.where(rho665 < self.switch_threshold, spm_665, spm_865)
        return np.where(result > 0, result, np.nan).astype(np.float32)


@register_algorithm('spm', 'dogliotti2015')
class Dogliotti2015SPM(WQAlgorithm):
    """Switching red/NIR SPM algorithm derived from Dogliotti et al. (2015).

    First retrieves turbidity T [FNU] via the switching algorithm, then
    converts to SPM using the empirical power-law:
        SPM [g m⁻³] ≈ 1.24 · T^1.12

    The SPM→turbidity conversion is an approximation; for direct turbidity
    retrieval use the dogliotti2015_t algorithm in turbidity.py.
    """
    product = 'spm'
    name = 'dogliotti2015'
    units = 'g m-3'
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
        'spm_A':  1.24,   # SPM = spm_A * T^spm_B
        'spm_B':  1.12,
    }

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        rho665, rho865 = bands[665], bands[865]
        T = _dogliotti_t(rho665, rho865,
                         self.AT_red, self.CT_red,
                         self.AT_nir, self.CT_nir,
                         self.t_low, self.t_high)
        result = self.spm_A * np.power(np.where(T > 0, T, np.nan), self.spm_B)
        return np.where(result > 0, result, np.nan).astype(np.float32)


@register_algorithm('spm', 'doxaran2012')
class Doxaran2012SPM(WQAlgorithm):
    """Power-law NIR/VIS band-ratio SPM — calibrated on Mackenzie Arctic plume.

    SPM = A · (Rrs(865) / Rrs(560)) ^ B

    Coefficients from Doxaran et al. (2012) calibrated on Mackenzie River
    plume (Canadian Arctic). Mackenzie River particles (permafrost clay/silt)
    are the most analogous published dataset to Nelson/Hayes/La Grande rivers
    entering Hudson Bay. π cancels in the ratio → input_quantity = 'Rrs'.
    """
    product = 'spm'
    name = 'doxaran2012'
    units = 'g m-3'
    reference = ('Doxaran et al. (2012). Biogeosciences 9:3213–3229. '
                 'doi:10.5194/bg-9-3213-2012')
    required_bands = [560, 865]
    input_quantity = 'Rrs'

    _DEFAULTS = {'A': 1306.0, 'B': 1.28}

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        r560, r865 = bands[560], bands[865]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(r560 > 0, r865 / r560, np.nan)
            result = self.A * np.power(ratio, self.B)
        return np.where(result > 0, result, np.nan).astype(np.float32)


# Mabit-style single-band SPM, per instrument family:
#   band = nominal wavelength read; SPM = 10 ** (A · Rrs(band) ** B)
# MSI uses the 705 nm red-edge band, OLI the 665 nm red band (OLI has no
# red edge), each with its own calibration pair.
_MABIT_SPM = {
    FAMILY_MSI: {'band': 705, 'A': 17.0, 'B': 0.42},
    FAMILY_OLI: {'band': 665, 'A': 13.0, 'B': 0.52},
}


@register_algorithm('spm', 'mabit_powerlaw')
class MabitPowerLawSPM(WQAlgorithm):
    """Single-band log-power SPM used for MSI and OLI in the SAMBA L3 chain.

    SPM = 10 ^ ( A · Rrs(λ) ^ B )

    The band and the coefficient pair are chosen from the instrument:

        MSI  λ = 705 nm (red edge)   A = 17,  B = 0.42
        OLI  λ = 665 nm (red)        A = 13,  B = 0.52

    Landsat OLI has no red-edge band, so it falls back to the red band with a
    separately fitted pair rather than the MSI coefficients.  Restricted to
    those two families: no OLCI calibration of this form has been published,
    and OLCI scenes should use nechad2010, dogliotti2015, or doxaran2012.

    Coefficient overrides are given per family, e.g.
    ``params: {MSI_A: 17.5, OLI_B: 0.50}``.
    """
    product = 'spm'
    name = 'mabit_powerlaw'
    units = 'g m-3'
    reference = ('Mabit et al. (2022). Frontiers in Remote Sensing 3:834908. '
                 'doi:10.3389/frsen.2022.834908 (single-band form, SAMBA L3 chain)')
    required_bands = {
        FAMILY_MSI: [_MABIT_SPM[FAMILY_MSI]['band']],
        FAMILY_OLI: [_MABIT_SPM[FAMILY_OLI]['band']],
    }
    families = (FAMILY_MSI, FAMILY_OLI)
    input_quantity = 'Rrs'

    _DEFAULTS = {
        'MSI_A': _MABIT_SPM[FAMILY_MSI]['A'],
        'MSI_B': _MABIT_SPM[FAMILY_MSI]['B'],
        'OLI_A': _MABIT_SPM[FAMILY_OLI]['A'],
        'OLI_B': _MABIT_SPM[FAMILY_OLI]['B'],
    }

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        family = self.family
        if family not in _MABIT_SPM:
            raise ValueError(
                f'{self.name} has no calibration for family {family!r}; '
                f'available: {sorted(_MABIT_SPM)}'
            )
        rrs = bands[_MABIT_SPM[family]['band']]
        A = getattr(self, f'{family}_A')
        B = getattr(self, f'{family}_B')
        with np.errstate(divide='ignore', invalid='ignore'):
            # Rrs must be positive: a fractional power of a negative Rrs is undefined.
            base = np.where(rrs > 0, rrs, np.nan)
            result = np.power(10.0, A * np.power(base, B))
        return np.where(result > 0, result, np.nan).astype(np.float32)
