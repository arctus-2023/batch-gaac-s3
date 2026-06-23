"""WQConfig — configuration dataclass and YAML loader."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WQConfig:
    l2_dir: str
    l3_dir: str
    aoi_name: str
    wq_products: dict[str, dict[str, Any]]
    aggregation: dict[str, Any]
    gaac_gen_dir: str | None = None
    replace_output: bool = False


def load_config(path: str | Path) -> WQConfig:
    """Load and validate a WQ pipeline YAML config file."""
    with open(path) as f:
        raw: dict = yaml.safe_load(f)

    required = ['l2_dir', 'l3_dir', 'aoi_name', 'wq_products']
    for key in required:
        if key not in raw:
            raise ValueError(f'Config missing required key: {key!r}')

    # Normalize each product entry
    products: dict[str, dict[str, Any]] = {}
    for prod_name, prod_cfg in raw['wq_products'].items():
        if not isinstance(prod_cfg, dict):
            raise ValueError(
                f'wq_products.{prod_name} must be a dict, got {type(prod_cfg)}'
            )
        if 'algorithm' not in prod_cfg:
            raise ValueError(
                f'wq_products.{prod_name} is missing required key "algorithm"'
            )
        products[prod_name] = {
            'enabled':   prod_cfg.get('enabled', True),
            'algorithm': prod_cfg['algorithm'],
            'params':    prod_cfg.get('params') or {},
        }

    aggregation = raw.get('aggregation', {
        'method':  'mean',
        'periods': ['daily', 'monthly', 'yearly'],
    })

    return WQConfig(
        l2_dir=str(raw['l2_dir']),
        l3_dir=str(raw['l3_dir']),
        aoi_name=str(raw['aoi_name']),
        wq_products=products,
        aggregation=aggregation,
        gaac_gen_dir=raw.get('gaac_gen_dir'),
        replace_output=bool(raw.get('replace_output', False)),
    )
