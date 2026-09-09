"""Abstract base class for all water quality retrieval algorithms."""

from __future__ import annotations
import abc
import numpy as np

from ..sensors import UNKNOWN, SensorSpec


class WQAlgorithm(abc.ABC):
    """Base class for water quality retrieval algorithms.

    Subclass contract
    -----------------
    - Declare class attributes: product, name, units, reference, required_bands
    - Set input_quantity = 'Rrs' (default) or 'rhow'
    - Optionally restrict `families` to the instruments the algorithm is
      calibrated for; leave it None to accept any sensor that carries the
      required bands
    - Implement compute()
    - Declare _DEFAULTS dict with default coefficient values
    - Accept **params in __init__ to override those defaults

    Multi-sensor notes
    ------------------
    `required_bands` holds *nominal* wavelengths — the OLCI-centric numbers used
    in the literature.  SceneProcessor resolves each of them to the nearest band
    the scene actually carries (see `sensors.resolve_wavelengths`) and re-keys
    the arrays back onto the nominal values, so `compute()` can index
    `bands[665]` whether the file stores 665 nm (OLCI/MSI) or 655 nm (OLI).

    An algorithm whose coefficients or band choice differ per instrument may
    declare `required_bands` as a dict keyed by family, and read `self.family`
    inside `compute()` — SceneProcessor calls `bind()` with the scene's sensor
    before every retrieval.
    """

    # --- class-level declarations (subclasses must override) ---
    product: str           # 'chla' | 'cdom' | 'spm' | 'turbidity'
    name: str              # algorithm identifier, e.g. 'gons2005'
    units: str             # physical units, e.g. 'mg m-3'
    reference: str         # short citation

    # Nominal wavelengths (nm int) this algorithm reads from the bands dict.
    # Either a flat list (same for every instrument) or {family: [wavelengths]}
    # when the algorithm reads different bands on different instruments.
    required_bands: list[int] | dict[str, list[int]]

    # Instrument families this algorithm is calibrated for ('OLCI', 'MSI',
    # 'OLI'). None = run on any sensor that carries the required bands.
    families: tuple[str, ...] | None = None

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
        self.sensor: SensorSpec = UNKNOWN

    # ── sensor binding ────────────────────────────────────────────────────────

    def bind(self, sensor: SensorSpec) -> None:
        """Attach the scene's sensor before `compute()`; see `self.family`."""
        self.sensor = sensor

    @property
    def family(self) -> str:
        """Instrument family of the scene currently bound ('OLCI'/'MSI'/'OLI')."""
        return self.sensor.family

    @classmethod
    def supports(cls, sensor: SensorSpec) -> bool:
        """True when this algorithm is calibrated for `sensor`'s family."""
        return cls.families is None or sensor.family in cls.families

    @classmethod
    def bands_for(cls, family: str) -> list[int]:
        """Nominal wavelengths this algorithm needs on the given family."""
        if isinstance(cls.required_bands, dict):
            return list(cls.required_bands.get(family, []))
        return list(cls.required_bands)

    @abc.abstractmethod
    def compute(self, bands: dict[int, np.ndarray]) -> np.ndarray:
        """Retrieve the water quality variable.

        Parameters
        ----------
        bands : dict[int, np.ndarray]
            Arrays keyed by the *nominal* wavelength (nm) declared in
            `required_bands`, regardless of the sensor's actual band centre.
            Values are float32, shape (H, W).  Non-water pixels are already NaN.
            Units depend on self.input_quantity: Rrs [sr⁻¹] or ρw [dimensionless].

        Returns
        -------
        np.ndarray, float32, shape (H, W)
            Retrieved product values. NaN where computation is invalid
            (division by zero, physically impossible result, etc.).
        """
