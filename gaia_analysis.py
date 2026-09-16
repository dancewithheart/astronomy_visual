#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from gaia_orion_flythrough import QueryConfig, add_derived_columns, get_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path("analysis")
REPORT_FILE = OUTPUT_DIR / "gaia-analysis.md"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data available._"

    display = frame.fillna("").astype(str)

    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    headers = [cell(str(column)) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def prepare_data(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = QueryConfig()
    raw = get_data(config, refresh=refresh)

    required = {
        "ra",
        "dec",
        "parallax",
        "parallax_over_error",
        "bp_rp",
        "phot_g_mean_mag",
    }
    missing = required.difference(raw.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Cached Gaia data is missing {names}. Run `python gaia_analysis.py --refresh`."
        )

    numeric = raw.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    clean = numeric.dropna(subset=list(required))
    clean = clean[(clean["parallax"] > 0) & (clean["parallax_over_error"] >= 5)].copy()
    clean = add_derived_columns(clean, config.ra_deg, config.dec_deg)
    return raw, clean


def distance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 200, 400, 600, 800, 1000, float("inf")]
    labels = ["0–200", "200–400", "400–600", "600–800", "800–1000", "1000+"]
    distance_band = pd.cut(
        frame["distance_pc"],
        bins=bins,
        labels=labels,
        right=False,
    )
    return (
        frame.assign(distance_band=distance_band)
        .groupby("distance_band", observed=False)
        .agg(
            stars=("source_id", "count"),
            median_distance_pc=("distance_pc", "median"),
            median_bp_rp=("bp_rp", "median"),
            median_g_mag=("phot_g_mean_mag", "median"),
            median_parallax_over_error=("parallax_over_error", "median"),
        )
        .reset_index()
        .round(2)
    )


def save_plots(frame: pd.DataFrame) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    distance_path = OUTPUT_DIR / "distance-histogram.png"
    axis = frame["distance_pc"].hist(bins=35, figsize=(8, 5), color="#4C78A8")
    axis.set(title="Gaia source distance distribution", xlabel="Distance (pc)", ylabel="Stars")
    axis.figure.tight_layout()
    axis.figure.savefig(distance_path, dpi=160)
    plt.close(axis.figure)

    color_path = OUTPUT_DIR / "distance-vs-color.png"
    fig, axis = plt.subplots(figsize=(8, 5))
    points = axis.scatter(
        frame["distance_pc"],
        frame["bp_rp"],
        c=frame["phot_g_mean_mag"],
        cmap="viridis_r",
        alpha=0.35,
        s=9,
    )
    axis.set(title="Distance versus Gaia BP-RP color", xlabel="Distance (pc)", ylabel="BP-RP")
    fig.colorbar(points, ax=axis, label="G apparent magnitude")
    fig.tight_layout()
    fig.savefig(color_path, dpi=160)
    plt.close(fig)

    quality_path = OUTPUT_DIR / "parallax-quality.png"
    axis = frame["parallax_over_error"].clip(upper=100).hist(
        bins=35,
        figsize=(8, 5),
        color="#59A14F",
    )
    axis.set(
        title="Parallax measurement quality",
        xlabel="Parallax / parallax error (clipped at 100 for display)",
        ylabel="Stars",
    )
    axis.figure.tight_layout()
    axis.figure.savefig(quality_path, dpi=160)
    plt.close(axis.figure)

    return [distance_path, color_path, quality_path]


def build_report(raw: pd.DataFrame, clean: pd.DataFrame, plot_paths: list[Path]) -> None:
    summary_columns = [
        "parallax",
        "parallax_over_error",
        "distance_pc",
        "bp_rp",
        "phot_g_mean_mag",
    ]
    descriptive = (
        clean[summary_columns]
        .describe()
        .round(3)
        .rename_axis("statistic")
        .reset_index()
    )
    correlations = (
        clean[["distance_pc", "bp_rp", "phot_g_mean_mag", "parallax_over_error"]]
        .corr()
        .round(3)
        .rename_axis("variable")
        .reset_index()
    )
    bands = distance_summary(clean)

    relative_paths = [path.relative_to(OUTPUT_DIR) for path in plot_paths]
    lines = [
        "# Gaia Orion exploratory analysis",
        "",
        "The stars and measurements in this report come from Gaia DR3. The analysis is "
        "descriptive and does not validate the synthetic nebula rendering.",
        "",
        "## Data quality",
        "",
        f"- queried/cached rows: **{len(raw):,}**",
        f"- rows after complete-case and `parallax_over_error >= 5` filtering: **{len(clean):,}**",
        "",
        "## Descriptive statistics",
        "",
        markdown_table(descriptive),
        "",
        f"![Distance histogram]({relative_paths[0].as_posix()})",
        "",
        "## Distance bands",
        "",
        markdown_table(bands),
        "",
        f"![Distance versus color]({relative_paths[1].as_posix()})",
        "",
        "## Correlations",
        "",
        markdown_table(correlations),
        "",
        "Correlation is descriptive and affected by query truncation, quality filters, "
        "apparent-magnitude selection, and inverse-parallax distance estimation.",
        "",
        f"![Parallax quality]({relative_paths[2].as_posix()})",
        "",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cached Gaia Orion data")
    parser.add_argument("--refresh", action="store_true", help="Run a new Gaia query")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, clean = prepare_data(refresh=args.refresh)
    plot_paths = save_plots(clean)
    build_report(raw, clean, plot_paths)
    print(f"wrote analysis report to {REPORT_FILE}")


if __name__ == "__main__":
    main()
