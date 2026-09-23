"""named astronomy datasets"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gaia import QueryConfig, query_gaia


DATA_DIR = Path("data")


@dataclass(frozen=True)
class GaiaDataset:
    name: str
    query: QueryConfig

    @property
    def cache_path(self) -> Path:
        return DATA_DIR / f"{self.name}.csv"


ORION = GaiaDataset(
    name="orion",
    query=QueryConfig(
        ra_deg=83.82208,
        dec_deg=-5.39111,
        radius_deg=3.0,
        row_limit=50_000,
        g_mag_max=15.5,
        parallax_over_error_min=5.0,
        parallax_min=1.0,
        parallax_max=8.0,
        ruwe_max=1.4,
    )
)


PLEIADES = GaiaDataset(
    name="pleiades",
    query=QueryConfig(
        ra_deg=56.87125,
        dec_deg=24.10493,
        radius_deg=3.0,
        row_limit=50_000,
        parallax_over_error_min=10.0,
        ruwe_max=1.4,
    ),
)


def load_dataset(dataset: GaiaDataset, *, refresh: bool = False) -> pd.DataFrame:
    cache = dataset.cache_path

    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    data = query_gaia(dataset.query)

    DATA_DIR.mkdir(exist_ok=True)
    data.to_csv(cache, index=False)

    return data
