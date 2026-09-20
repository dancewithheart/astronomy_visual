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


def plot_proper_motion(data: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))

    plt.scatter(
        data["pmra"],
        data["pmdec"],
        c=data["cluster"],
        s=4,
        alpha=0.5,
    )

    plt.xlabel("pmra [mas/year]")
    plt.ylabel("pmdec [mas/year]")
    plt.title("Gaia proper motions around the Pleiades")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "proper-motion-clusters.png",
        dpi=160,
        )

    plt.close()


def plot_sky(data: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(9, 7))

    plt.scatter(
        data["ra"],
        data["dec"],
        c=data["cluster"],
        s=4,
        alpha=0.5,
    )

    plt.xlabel("Right ascension [deg]")
    plt.ylabel("Declination [deg]")
    plt.title("DBSCAN clusters on the sky")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "sky-clusters.png",
        dpi=160,
        )
    plt.close()


def plot_candidate_cmd(data: pd.DataFrame, cluster: int) -> None:
    candidate = data.loc[
        (data["cluster"] == cluster)
        & data["bp_rp"].notna()
        & data["phot_g_mean_mag"].notna()
        ]

    plt.figure(figsize=(7, 8))

    plt.scatter(
        candidate["bp_rp"],
        candidate["phot_g_mean_mag"],
        s=8,
        alpha=0.7,
    )

    plt.xlabel("BP - RP")
    plt.ylabel("G magnitude")
    plt.title(
        f"Colour-magnitude diagram: cluster {cluster}"
    )

    # Brighter stars conventionally appear at the top.
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "candidate-cmd.png",
        dpi=160,
        )

    plt.close()

def plot_cmd(data: pd.DataFrame) -> None:
    valid = data[
        data["bp_rp"].notna()
        & data["phot_g_mean_mag"].notna()
        ]

    candidate = valid[valid["cluster"] >= 0]
    noise = valid[valid["cluster"] == -1]

    plt.figure(figsize=(7, 8))

    plt.scatter(
        noise["bp_rp"],
        noise["phot_g_mean_mag"],
        s=8,
        alpha=0.25,
        label="field / noise",
    )

    plt.scatter(
        candidate["bp_rp"],
        candidate["phot_g_mean_mag"],
        s=10,
        alpha=0.7,
        label="DBSCAN candidate",
    )

    plt.xlabel("BP - RP")
    plt.ylabel("G magnitude")
    plt.title("Pleiades candidate vs field stars")
    plt.gca().invert_yaxis()
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "cmd-candidate-vs-field.png",
        dpi=160,
        )
    plt.close()

def plot_angular_distance(data: pd.DataFrame) -> None:
    candidate = data[data["cluster"] >= 0]
    noise = data[data["cluster"] == -1]

    plt.figure(figsize=(8, 6))

    plt.hist(
        noise["angular_distance_deg"],
        bins=30,
        alpha=0.4,
        label="field / noise",
    )

    plt.hist(
        candidate["angular_distance_deg"],
        bins=30,
        alpha=0.7,
        label="DBSCAN candidate",
    )

    plt.xlabel("Angular distance from Pleiades centre [deg]")
    plt.ylabel("Stars")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "angular-distance.png",
        dpi=160,
        )
    plt.close()

def main(*, refresh: bool, eps: float, min_samples: int) -> None:
    raw = load_dataset(PLEIADES, refresh=refresh)

    print(f"Downloaded stars: {len(raw):,}")
    prepared = prepare_features(raw)
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

    plot_proper_motion(clustered)
    plot_sky(clustered)

    if not summary.empty:
        candidate_cluster = int(summary.index[0])
        print("Largest dense cluster:", candidate_cluster)
        plot_candidate_cmd(clustered, candidate_cluster)

    plot_cmd(clustered)
    plot_parallax(clustered)
    plot_angular_distance(clustered)


def plot_parallax(data: pd.DataFrame) -> None:
    candidate = data[data["cluster"] >= 0]
    noise = data[data["cluster"] == -1]

    plt.figure(figsize=(8, 6))

    plt.hist(
        noise["parallax"],
        bins=40,
        alpha=0.4,
        label="field / noise",
    )

    plt.hist(
        candidate["parallax"],
        bins=40,
        alpha=0.7,
        label="DBSCAN candidate",
    )

    plt.xlabel("Parallax [mas]")
    plt.ylabel("Stars")
    plt.title("Parallax distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "parallax.png",
        dpi=160,
        )
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--min-samples", type=int, default=15)
    args = parser.parse_args()

    main(refresh=args.refresh, eps=args.eps, min_samples=args.min_samples)
