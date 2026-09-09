"""Sensor identification and nominal → actual band resolution.

The GAAC L2 writer stamps a ``sensor`` tag on every ``*_rhor_rhow.tif``
(``S3_OLCIA``, ``S2_MSIB``, ``L8_OLI``, ``L9_OLI2``, …).  That tag is the
primary source of truth here; the scene filename is used as a fallback for
products written before the tag existed.

Sensors are grouped into three *families* because water-quality algorithms
are calibrated per instrument type, not per platform:

    OLCI : Sentinel-3A/B OLCI    (400–1020 nm, 15 water bands in the L2 file)
    MSI  : Sentinel-2A/B/C MSI   (442–864 nm, 9 water bands)
    OLI  : Landsat-8 OLI / 9 OLI-2 (443–865 nm, 5 water bands)

Band resolution
---------------
Algorithms declare *nominal* wavelengths (the OLCI-centric numbers used in the
literature, e.g. 665 / 709 / 865 nm).  ``resolve_wavelengths`` maps each of
those onto the nearest band the scene actually carries, provided it lies within
``tolerance`` nm — so an algorithm asking for 665 nm gets OLI's 655 nm band and
one asking for 709 nm gets MSI's 704 nm band, while a request that has no
counterpart (e.g. 510 nm on OLI) is reported as missing and the algorithm is
skipped for that scene.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── families ──────────────────────────────────────────────────────────────────

FAMILY_OLCI = 'OLCI'
FAMILY_MSI  = 'MSI'
FAMILY_OLI  = 'OLI'

FAMILIES = (FAMILY_OLCI, FAMILY_MSI, FAMILY_OLI)

#: Default nominal↔actual matching tolerance [nm].
DEFAULT_TOLERANCE = 15


@dataclass(frozen=True)
class SensorSpec:
    """One satellite instrument.

    key          : the ``sensor`` tag value written by gaac, e.g. ``'L8_OLI'``
    family       : instrument, ``'OLCI'`` | ``'MSI'`` | ``'OLI'``
    label        : short platform label, e.g. ``'L8'``, ``'S2B'``
    constellation: mission series, e.g. ``'S2'`` (covers S2A/S2B/S2C),
                   ``'S3'``, ``'LANDSAT'`` — so a selector like ``S2`` picks up
                   every Sentinel-2 platform without naming each one
    """
    key: str
    family: str
    label: str
    constellation: str


_SPECS = (
    SensorSpec('S3_OLCIA', FAMILY_OLCI, 'S3A', 'S3'),
    SensorSpec('S3_OLCIB', FAMILY_OLCI, 'S3B', 'S3'),
    SensorSpec('S2_MSIA',  FAMILY_MSI,  'S2A', 'S2'),
    SensorSpec('S2_MSIB',  FAMILY_MSI,  'S2B', 'S2'),
    SensorSpec('S2_MSIC',  FAMILY_MSI,  'S2C', 'S2'),
    SensorSpec('L8_OLI',   FAMILY_OLI,  'L8',  'LANDSAT'),
    SensorSpec('L9_OLI2',  FAMILY_OLI,  'L9',  'LANDSAT'),
)

#: {sensor tag: SensorSpec}
SENSORS: dict[str, SensorSpec] = {s.key: s for s in _SPECS}

#: Used when the platform is unknown but the instrument family is not
#: (e.g. a scene named ``S2_L1TOA_...`` with no sensor tag).
_GENERIC = {
    FAMILY_OLCI: SensorSpec('S3_OLCI', FAMILY_OLCI, 'S3', 'S3'),
    FAMILY_MSI:  SensorSpec('S2_MSI',  FAMILY_MSI,  'S2', 'S2'),
    FAMILY_OLI:  SensorSpec('L_OLI',   FAMILY_OLI,  'L',  'LANDSAT'),
}

UNKNOWN = SensorSpec('UNKNOWN', 'UNKNOWN', 'UNK', 'UNKNOWN')

# Filename prefixes, most specific first. Matched against the upper-cased stem.
_NAME_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r'(?:^|[_-])S3A'),          'S3_OLCIA'),
    (re.compile(r'(?:^|[_-])S3B'),          'S3_OLCIB'),
    (re.compile(r'(?:^|[_-])S3'),           'S3_OLCI'),
    (re.compile(r'(?:^|[_-])S2A'),          'S2_MSIA'),
    (re.compile(r'(?:^|[_-])S2B'),          'S2_MSIB'),
    (re.compile(r'(?:^|[_-])S2C'),          'S2_MSIC'),
    (re.compile(r'(?:^|[_-])S2'),           'S2_MSI'),
    (re.compile(r'(?:^|[_-])LC0?8'),        'L8_OLI'),
    (re.compile(r'(?:^|[_-])LC0?9'),        'L9_OLI2'),
    (re.compile(r'(?:^|[_-])L8'),           'L8_OLI'),
    (re.compile(r'(?:^|[_-])L9'),           'L9_OLI2'),
)

_ALL_BY_KEY: dict[str, SensorSpec] = {
    **SENSORS,
    **{s.key: s for s in _GENERIC.values()},
}


def sensor_from_key(key: str | None) -> SensorSpec | None:
    """Return the spec for an exact sensor tag, or infer its family from a prefix."""
    if not key:
        return None
    key = key.strip()
    if key in _ALL_BY_KEY:
        return _ALL_BY_KEY[key]
    # Unlisted platform of a known instrument, e.g. a future 'S2D_MSID'
    upper = key.upper()
    for family, marker in ((FAMILY_OLCI, 'OLCI'), (FAMILY_MSI, 'MSI'), (FAMILY_OLI, 'OLI')):
        if marker in upper:
            return _GENERIC[family]
    return None


def generic_sensor(family: str) -> SensorSpec | None:
    """A placeholder spec standing for a whole instrument family.

    Used where only the family is known — DRP aggregation, which composites
    across the platforms of one instrument, and `--sensor OLI`-style filters.
    """
    return _GENERIC.get(family.upper())


def sensor_from_name(name: str) -> SensorSpec | None:
    """Infer the sensor from a scene / file name, e.g. ``'LC08_L1TOA_20250628_…'``."""
    upper = name.upper()
    for pattern, key in _NAME_PATTERNS:
        if pattern.search(upper):
            return _ALL_BY_KEY[key]
    return None


def detect_sensor(tag: str | None = None, name: str = '') -> SensorSpec:
    """Resolve a scene's sensor from its ``sensor`` tag, falling back to its name.

    Returns :data:`UNKNOWN` when neither source is conclusive; algorithms that
    do not restrict themselves to a family still run in that case.
    """
    return sensor_from_key(tag) or sensor_from_name(name) or UNKNOWN


def matches(spec: SensorSpec, selectors: list[str] | None) -> bool:
    """True when `spec` is named by any selector, or when there are no selectors.

    A selector may be an exact sensor tag (``L8_OLI``), a platform label
    (``L8``, ``S2B``), a mission series (``S2``, ``S3``, ``LANDSAT``), an
    instrument family (``OLI``, ``MSI``, ``OLCI``), or ``all``.
    """
    if not selectors:
        return True
    names = {spec.key.upper(), spec.label.upper(), spec.family.upper(),
             spec.constellation.upper()}
    return any(sel.upper() in names or sel.upper() == 'ALL' for sel in selectors)


def resolve_wavelengths(
    available: list[int] | set[int],
    requested: list[int],
    tolerance: int = DEFAULT_TOLERANCE,
) -> tuple[dict[int, int], list[int]]:
    """Map nominal wavelengths onto the nearest bands the scene actually has.

    Parameters
    ----------
    available : wavelengths (nm) present in the scene
    requested : nominal wavelengths (nm) an algorithm asks for
    tolerance : maximum |nominal − actual| accepted [nm]

    Returns
    -------
    (mapping, missing)
        ``mapping`` is ``{nominal: actual}`` for every resolvable band;
        ``missing`` lists the nominal wavelengths with no band within tolerance.
    """
    pool = sorted(available)
    mapping: dict[int, int] = {}
    missing: list[int] = []

    for wl in requested:
        if not pool:
            missing.append(wl)
            continue
        best = min(pool, key=lambda a: abs(a - wl))
        if abs(best - wl) > tolerance:
            missing.append(wl)
        else:
            mapping[wl] = best

    return mapping, missing
