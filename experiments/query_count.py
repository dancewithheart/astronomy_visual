import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from datasets import PLEIADES, load_dataset


FEATURES = ["parallax", "pmra", "pmdec"]

REPORT_DIR = Path("reports/pleiades")

PLEIADES_RA = 56.87125
PLEIADES_DEC = 24.10493


def add_angular_distance(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    dra = (result["ra"] - PLEIADES_RA) * np.cos(np.radians(PLEIADES_DEC))
    ddec = result["dec"] - PLEIADES_DEC
    result["angular_distance_deg"] = np.sqrt(dra**2 + ddec**2)
    return result


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .dropna(subset=FEATURES)
        .loc[lambda df: df["parallax"] > 0]
        .copy()
    )


def cluster_stars(
        data: pd.DataFrame,
        *,
        eps: float = 0.25,
        min_samples: int = 15,
) -> pd.DataFrame:
    result = data.copy()
    scaler = StandardScaler()
    features = scaler.fit_transform(result[FEATURES])
    model = DBSCAN(eps=eps, min_samples=min_samples)
    result["cluster"] = model.fit_predict(features)
    return result

def summarize_clusters(data: pd.DataFrame) -> pd.DataFrame:
    clusters = data.loc[data["cluster"] >= 0]

    return (
        clusters
        .groupby("cluster")
        .agg(
            stars=("source_id", "size"),
            parallax_mas=("parallax", "median"),
            pmra_mas_yr=("pmra", "median"),
            pmdec_mas_yr=("pmdec", "median"),
        )
        .sort_values(
            "stars",
            ascending=False,
        )
    )


def add_absolute_g_magnitude(
        data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    distance_pc = 1000.0 / result["parallax"]

    result["absolute_g_mag"] = (
            result["phot_g_mean_mag"]
            - 5 * np.log10(distance_pc)
            + 5
    )

    return result

def main(*, refresh: bool, eps: float, min_samples: int) -> None:
    raw = load_dataset(PLEIADES, refresh=refresh)

    print(f"Downloaded stars: {len(raw):,}")
    prepared = prepare_features(raw)
    prepared = add_absolute_g_magnitude(prepared)
    print(
        f"Stars after preparation: "
        f"{len(prepared):,}"
    )

    clustered = cluster_stars(prepared, eps=eps, min_samples=min_samples)
    clustered = add_angular_distance(clustered)

    summary = summarize_clusters(clustered)
    print()
    print(summary)

    noise = (clustered["cluster"] == -1).sum()
    print()
    print(f"Noise stars: {noise:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--min-samples", type=int, default=15)
    args = parser.parse_args()

    main(refresh=args.refresh, eps=args.eps, min_samples=args.min_samples)
