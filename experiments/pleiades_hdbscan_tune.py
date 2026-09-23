import csv

import experiments.pleiades_hdbscan as pleiades

if __name__ == "__main__":
    rows = []

    for min_cluster_size in [20, 50, 100, 200]:
        for min_samples in [10, 15, 20, 30]:
            print("==================================")
            row = pleiades.main(min_cluster_size=min_cluster_size, min_samples=min_samples)
            rows.append(row)
            print(row)

    with open("reports/pleiades/hdbscan-tuning.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
