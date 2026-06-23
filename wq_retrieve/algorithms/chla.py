"""Chlorophyll-a retrieval algorithms [mg m⁻³].

Algorithms
----------
oc4me    : OC4Me band-ratio polynomial — ESA OLCI operational product
           O'Reilly et al. 1998 / ESA ATBD 2013
gons2005 : NIR-red semi-analytical (CDOM-insensitive)
           Gons et al. 2005, J. Plankton Research
ndci     : Normalized Difference Chlorophyll Index
           Mishra & Mishra 2012, Remote Sensing of Environment
"""

from __future__ import annotations
import numpy as np
from ..registry import register_algorithm
from .base import WQAlgorithm

# Pure-water absorption coefficients at OLCI red/NIR wavelengths (Pope & Fry 1997)
_AW_665 = 0.401    # m⁻¹
_AW_709 = 0.703    # m⁻¹

# Chlorophyll-specific absorption at 665 nm (Gons 1999 / Gons et al. 2005)
_APH_STAR_665 = 0.0149   # m² mg⁻¹


@register_algorithm('chla', 'oc4me')
class OC4MeChla(WQAlgorithm):
    """OC4Me — ESA OLCI operational chlorophyll algorithm.

    log10(Chla) = A0 + A1·X + A2·X² + A3·X³ + A4·X⁴
    X = log10[ max(Rrs443, Rrs490, Rrs510) / Rrs560 ]

    Coefficients from ESA OLCI L2 ATBD (ACRI-ST, 2013).

    NOTE: OC4Me overestimates Chla 2–10× in CDOM-rich Arctic/subarctic
    nearshore waters. Use gons2005 or ndci for river-plume pixels.
    """
    product = 'chla'
    name = 'oc4me'
    units = 'mg m-3'
    reference = ("O'Reilly et al. (1998). J. Geophys. Res. 103(C11):24937. "
                 "ESA OLCI L2 ATBD coefficients (2013).")
    required_bands = [443, 490, 510, 560]
    input_quantity = 'Rrs'

    _DEFAULTS = {
        'A0':  0.4503,
        'A1': -3.2595,
        'A2':  3.5227,
        'A3': -3.3594,
        'A4':  0.9496,
    }

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        r443, r490, r510, r560 = bands[443], bands[490], bands[510], bands[560]
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = np.where(
                np.isfinite(r443) & np.isfinite(r490) & np.isfinite(r510),
                np.fmax(np.fmax(r443, r490), r510),
                np.nan,
            )
            ratio = np.where(r560 > 0, numerator / r560, np.nan)
            X = np.where(ratio > 0, np.log10(ratio), np.nan)
            log_chla = (self.A0 + self.A1 * X + self.A2 * X**2
                        + self.A3 * X**3 + self.A4 * X**4)
            result = np.power(10.0, log_chla)
        return np.where(result > 0, result, np.nan).astype(np.float32)


@register_algorithm('chla', 'gons2005')
class Gons2005Chla(WQAlgorithm):
    """NIR-red semi-analytical Chla retrieval (CDOM-insensitive).

    bbp  = 1.61 · Rrs(779) / (0.082 − 0.6 · Rrs(779))
    Chla = [ (Rrs(709)/Rrs(665)) · aw(665) − aw(709) + bbp ] / aph*(665)

    Operates entirely at 665–779 nm where CDOM absorption is negligible →
    immune to blue-band CDOM contamination. Recommended for James Bay / Hudson
    Bay nearshore pixels.
    """
    product = 'chla'
    name = 'gons2005'
    units = 'mg m-3'
    reference = ('Gons et al. (2005). J. Plankton Research 27(2):125–133. '
                 'doi:10.1093/plankt/fbh151')
    required_bands = [665, 709, 779]
    input_quantity = 'Rrs'

    _DEFAULTS = {
        'aw_665':    _AW_665,
        'aw_709':    _AW_709,
        'aph_star':  _APH_STAR_665,
    }

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        r665, r709, r779 = bands[665], bands[709], bands[779]
        with np.errstate(divide='ignore', invalid='ignore'):
            denom_bbp = 0.082 - 0.6 * r779
            bbp = np.where(denom_bbp > 0, 1.61 * r779 / denom_bbp, np.nan)
            ratio = np.where(r665 > 0, r709 / r665, np.nan)
            numer = ratio * self.aw_665 - self.aw_709 + bbp
            result = numer / self.aph_star
        return np.where(result > 0, result, np.nan).astype(np.float32)


@register_algorithm('chla', 'ndci')
class NDCIChla(WQAlgorithm):
    """Normalized Difference Chlorophyll Index.

    NDCI = (Rrs(709) − Rrs(665)) / (Rrs(709) + Rrs(665))
    Chla = A0 + A1·NDCI + A2·NDCI²

    CDOM-insensitive (both bands in red). Recommended for spring blooms
    and eutrophic river-plume conditions in James Bay.
    Calibration coefficients from Mishra & Mishra (2012) Table 1.
    """
    product = 'chla'
    name = 'ndci'
    units = 'mg m-3'
    reference = ('Mishra & Mishra (2012). Remote Sensing of Environment 117:394–406. '
                 'doi:10.1016/j.rse.2011.10.016')
    required_bands = [665, 709]
    input_quantity = 'Rrs'

    _DEFAULTS = {'A0': 14.039, 'A1': 86.115, 'A2': 194.325}

    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        r665, r709 = bands[665], bands[709]
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = r709 + r665
            ndci = np.where(denom > 0, (r709 - r665) / denom, np.nan)
            result = self.A0 + self.A1 * ndci + self.A2 * ndci**2
        return np.where(result > 0, result, np.nan).astype(np.float32)
