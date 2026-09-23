from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from datasets import PLEIADES, load_dataset
from experiments.pleiades import (
    FEATURES,
    add_absolute_g_magnitude,
    add_angular_distance,
    evaluate_pleiades_cluster,
    load_reference_pleiades,
    summarize_clusters,
)


def cluster_stars_hdbscan(data, *, min_cluster_size: int = 50, min_samples: int = 15):
    result = data.copy()

    features = StandardScaler().fit_transform(
        result[FEATURES]
    )

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        copy=True
    )

    result["cluster"] = model.fit_predict(features)

    return result


def main(*, min_cluster_size: int = 50, min_samples: int = 15):
    raw = load_dataset(PLEIADES, refresh=False)

    prepared = add_absolute_g_magnitude(raw)

    clustered = cluster_stars_hdbscan(
        prepared,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    clustered = add_angular_distance(clustered)

    summary = summarize_clusters(clustered)
    print(summary)

    reference = load_reference_pleiades()

    reference_ids = set(reference["source_id"].astype("int64"))

    evaluation = evaluate_pleiades_cluster(clustered, reference_ids)

    print()
    print(evaluation)

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "clusters": len(summary),
        "noise": int(
            (clustered["cluster"] < 0).sum()
        ),
        **evaluation,
    }


if __name__ == "__main__":
    print(main())
