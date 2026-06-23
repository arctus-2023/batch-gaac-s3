"""Abstract base class for all water quality retrieval algorithms."""

from __future__ import annotations
import abc
import numpy as np


class WQAlgorithm(abc.ABC):
    """Base class for water quality retrieval algorithms.

    Subclass contract
    -----------------
    - Declare class attributes: product, name, units, reference, required_bands
    - Set input_quantity = 'Rrs' (default) or 'rhow'
    - Implement compute()
    - Declare _DEFAULTS dict with default coefficient values
    - Accept **params in __init__ to override those defaults
    """

    # --- class-level declarations (subclasses must override) ---
    product: str           # 'chla' | 'cdom' | 'spm' | 'turbidity'
    name: str              # algorithm identifier, e.g. 'gons2005'
    units: str             # physical units, e.g. 'mg m-3'
    reference: str         # short citation

    # wavelengths (nm int) this algorithm reads from the bands dict
    required_bands: list[int]

    # 'Rrs' → SceneProcessor passes scene.rrs (= rhow/π)
    # 'rhow' → SceneProcessor passes scene.rhow (= ρw, dimensionless reflectance)
    # Band-ratio algorithms (NDCI, OC4Me, CDOM ratios): Rrs — π cancels in ratio
    # Absolute-value algorithms (Nechad, Dogliotti): rhow — calibrated to ρw
    input_quantity: str = 'Rrs'

    _DEFAULTS: dict = {}

    def __init__(self, **params: float) -> None:
        """Store coefficient overrides; fall back to _DEFAULTS for missing keys."""
        for key, default in self._DEFAULTS.items():
            setattr(self, key, params.get(key, default))

    @abc.abstractmethod
    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        """Retrieve the water quality variable.

        Parameters
        ----------
        bands : dict[int, np.ndarray]
            Arrays keyed by wavelength (nm). Values are float32, shape (H, W).
            Non-water pixels are already NaN.
            Units depend on self.input_quantity: Rrs [sr⁻¹] or ρw [dimensionless].

        Returns
        -------
        np.ndarray, float32, shape (H, W)
            Retrieved product values. NaN where computation is invalid
            (division by zero, physically impossible result, etc.).
        """
