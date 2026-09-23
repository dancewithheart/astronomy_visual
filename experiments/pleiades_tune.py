import csv

import experiments.pleiades as pleiades

if __name__ == "__main__":
    rows = []

    for eps in [0.001, 0.01, 0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for min_samples in [10, 15, 20, 30, 40, 50, 60, 70, 80, 120, 200, 400]:
            print("==================================")
            row = pleiades.main(refresh=False, eps=eps, min_samples=min_samples, make_plots=False)
            rows.append(row)
            print(row)

    with open("reports/pleiades/dbscan-tuning.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
