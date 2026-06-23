"""Algorithm registry — two-level dict keyed by (product, name).

Usage
-----
# Registering (in an algorithm module):
@register_algorithm('chla', 'gons2005')
class Gons2005Chla(WQAlgorithm): ...

# Look-up (in SceneProcessor):
AlgoCls = get_algorithm('chla', 'gons2005')
"""

from __future__ import annotations
from collections.abc import Callable

# { product_name: { algorithm_name: class } }
_REGISTRY: dict[str, dict[str, type]] = {}


def register_algorithm(product: str, name: str) -> Callable[[type], type]:
    """Class decorator that registers an algorithm under (product, name)."""
    def decorator(cls: type) -> type:
        _REGISTRY.setdefault(product, {})[name] = cls
        return cls
    return decorator


def get_algorithm(product: str, name: str) -> type:
    """Return the algorithm class for (product, name).

    Raises
    ------
    KeyError with a descriptive message listing available options.
    """
    if product not in _REGISTRY:
        raise KeyError(
            f"Unknown product '{product}'. "
            f"Available products: {sorted(_REGISTRY)}"
        )
    available = _REGISTRY[product]
    if name not in available:
        raise KeyError(
            f"Unknown algorithm '{name}' for product '{product}'. "
            f"Available: {sorted(available)}"
        )
    return available[name]


def list_algorithms() -> dict[str, list[str]]:
    """Return {product: [algorithm_name, ...]} for all registered algorithms."""
    return {product: sorted(algos) for product, algos in sorted(_REGISTRY.items())}
