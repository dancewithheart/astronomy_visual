import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from datasets import PLEIADES, load_dataset

from astroquery.vizier import Vizier

REFERENCE_CATALOG = "J/A+A/677/A163/members"


def load_reference_pleiades() -> pd.DataFrame:
    vizier = Vizier(
        columns=["GaiaDR3", "Cluster"],
        row_limit=-1,
    )

    tables = vizier.get_catalogs(REFERENCE_CATALOG)

    reference = tables[0].to_pandas()

    # VizieR strings can occasionally arrive as bytes.
    reference["Cluster"] = reference["Cluster"].map(
        lambda value:
        value.decode()
        if isinstance(value, bytes)
        else str(value)
    )

    pleiades = reference[
        reference["Cluster"].str.strip().str.casefold()
        == "pleiades"
        ].copy()

    pleiades = pleiades.rename(
        columns={"GaiaDR3": "source_id"}
    )

    pleiades["source_id"] = (
        pleiades["source_id"]
        .astype("int64")
    )

    return pleiades[["source_id"]].drop_duplicates()


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

    # Display limits only — does NOT affect DBSCAN/data.
    plt.xlim(-75, 100)
    plt.ylim(-125, 50)

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
        candidate["absolute_g_mag"],
        s=8,
        alpha=0.7,
    )

    plt.xlabel("BP - RP")
    plt.ylabel("Absolute G magnitude")
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

def plot_cmd(data: pd.DataFrame, candidate_cluster: int) -> None:
    valid = data[
        data["bp_rp"].notna()
        & data["absolute_g_mag"].notna()
        ]

    candidate = valid[valid["cluster"] == candidate_cluster]
    noise = valid[valid["cluster"] != candidate_cluster]

    plt.figure(figsize=(7, 8))

    plt.scatter(
        noise["bp_rp"],
        noise["absolute_g_mag"],
        s=8,
        alpha=0.25,
        label="field / noise",
    )

    plt.scatter(
        candidate["bp_rp"],
        candidate["absolute_g_mag"],
        s=10,
        alpha=0.7,
        label="DBSCAN candidate",
    )

    plt.xlabel("BP - RP")
    plt.ylabel("Absolute G magnitude")
    plt.title("Pleiades candidate vs field stars")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "cmd-candidate-vs-field.png",
        dpi=160,
        )
    plt.close()


def plot_radial_density(data: pd.DataFrame, candidate_cluster: int) -> None:
    candidate = data[data["cluster"] == candidate_cluster]
    noise = data[data["cluster"] != candidate_cluster]

    bins = np.linspace(0.0, PLEIADES.query.radius_deg,31)

    def density(stars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        counts, edges = np.histogram(
            stars["angular_distance_deg"],
            bins=bins,
        )

        annulus_area = np.pi * (
                edges[1:] ** 2 - edges[:-1] ** 2
        )

        centers = (edges[:-1] + edges[1:]) / 2

        return centers, counts / annulus_area

    candidate_r, candidate_density = density(candidate)
    field_r, field_density = density(noise)

    plt.figure(figsize=(8, 6))

    plt.plot(
        field_r,
        field_density,
        marker="o",
        label="field / noise",
    )

    plt.plot(
        candidate_r,
        candidate_density,
        marker="o",
        label="DBSCAN candidate",
    )

    plt.xlabel("Angular distance from Pleiades centre [deg]")
    plt.ylabel("Stars / deg²")
    plt.title("Radial stellar surface density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "radial-density.png",
        dpi=160,
        )
    plt.close()

def plot_angular_distance_cdf(data: pd.DataFrame, candidate_cluster: int) -> None:
    plt.figure(figsize=(8, 6))

    for mask, label in [
        (data["cluster"] == -1, "field / noise"),
        (data["cluster"] == candidate_cluster, "DBSCAN candidate"),
    ]:
        distances = np.sort(
            data.loc[mask, "angular_distance_deg"]
        )

        fraction = (
                np.arange(1, len(distances) + 1)
                / len(distances)
        )

        plt.plot(
            distances,
            fraction,
            label=label,
        )

    plt.xlabel("Angular distance from Pleiades centre [deg]")
    plt.ylabel("Fraction of stars")
    plt.title("Cumulative radial distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "angular-distance-cdf.png",
        dpi=160,
        )
    plt.close()


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

def evaluate_pleiades_cluster(data: pd.DataFrame, reference_ids: set[int]) -> dict[str, float | int]:
    result = data.copy()

    result["reference_member"] = (
        result["source_id"]
        .astype("int64")
        .isin(reference_ids)
    )

    clusters = result[result["cluster"] >= 0]

    overlap = (
        clusters[clusters["reference_member"]]
        .groupby("cluster")
        .size()
    )

    if overlap.empty:
        pleiades_cluster = -1
    else:
        pleiades_cluster = int(overlap.idxmax())

    predicted = result["cluster"] == pleiades_cluster
    truth = result["reference_member"]

    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    members = result[predicted]

    return {
        "pleiades_cluster": pleiades_cluster,
        "pleiades_cluster_size": len(members),
        "reference_members": len(reference_ids),
        "reference_in_sample": int(truth.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_parallax": members["parallax"].median(),
        "median_pmra": members["pmra"].median(),
        "median_pmdec": members["pmdec"].median(),
    }

def main(*, refresh: bool, eps: float, min_samples: int, make_plots: bool):
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

    if make_plots:
        plot_proper_motion(clustered)
        plot_sky(clustered)

    reference = load_reference_pleiades()

    reference_ids = set(reference["source_id"].astype("int64"))

    evaluation = evaluate_pleiades_cluster(clustered, reference_ids)

    candidate_cluster = int(evaluation["pleiades_cluster"])

    print("Pleiades candidate cluster:", candidate_cluster,)

    print()
    print("Published reference catalogue")
    print(
        f"  Pleiades members:       "
        f"{evaluation['reference_members']}"
    )
    print(
        f"  present in our sample:  "
        f"{evaluation['reference_in_sample']}"
    )

    print()
    print("Classification")
    print(f"  true positives:  {evaluation['tp']}")
    print(f"  false positives: {evaluation['fp']}")
    print(f"  false negatives: {evaluation['fn']}")
    print(f"  true negatives:  {evaluation['tn']}")

    print()
    print(
        f"  precision: "
        f"{evaluation['precision']:.3f}"
    )
    print(
        f"  recall:    "
        f"{evaluation['recall']:.3f}"
    )
    print(
        f"  F1:        "
        f"{evaluation['f1']:.3f}"
    )

    if candidate_cluster >= 0 and make_plots:
        plot_candidate_cmd(clustered, candidate_cluster)
        plot_cmd(clustered, candidate_cluster)
        plot_parallax(clustered, candidate_cluster)
        plot_angular_distance_cdf(clustered, candidate_cluster)
        plot_radial_density(clustered, candidate_cluster)
    return {
        "eps": eps,
        "min_samples": min_samples,
        "clusters": len(summary),
        "noise": int(noise),
        **evaluation,
    }

def plot_parallax(data: pd.DataFrame, candidate_cluster: int) -> None:
    candidate = data[data["cluster"] == candidate_cluster]
    noise = data[data["cluster"] != candidate_cluster]

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

    main(refresh=args.refresh, eps=args.eps, min_samples=args.min_samples, make_plots=True)
