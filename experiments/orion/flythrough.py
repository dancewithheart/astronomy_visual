#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from astrometry import add_local_cartesian
from datasets import ORION, load_dataset

REPORT_DIR = Path("reports/orion")
HTML_FILE = REPORT_DIR / "gaia_orion_local_3d.html"


def prepare_orion_data(data: pd.DataFrame) -> pd.DataFrame:
    result = add_local_cartesian(
        data,
        center_ra_deg=ORION.query.ra_deg,
        center_dec_deg=ORION.query.dec_deg,
    )

    result["bp_rp_clamped"] = result["bp_rp"].clip(-0.5, 4.0)

    brightness = 10 ** (-0.4 * result["phot_g_mean_mag"].to_numpy())
    brightness /= np.nanpercentile(brightness, 99.5)

    result["size"] = (
            0.35
            + 1.5 * np.clip(np.sqrt(brightness), 0, 1.0)
    )

    result["hover"] = (
            "Gaia DR3 source: "
            + result["source_id"].astype(str)
            + "<br>G mag: "
            + result["phot_g_mean_mag"].round(2).astype(str)
            + "<br>BP-RP: "
            + result["bp_rp"].round(2).astype(str)
            + "<br>Distance: "
            + result["distance_pc"].round(1).astype(str)
            + " pc"
    )

    return result


def make_nebula_points(
        df: pd.DataFrame,
        n_clouds: int = 4,
        points_per_cloud: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)

    anchors = df.nsmallest(max(80, n_clouds * 20), "phot_g_mean_mag").copy()
    anchors = anchors[np.abs(anchors["z"]) < np.nanpercentile(np.abs(anchors["z"]), 70)]
    if len(anchors) < n_clouds:
        anchors = df.nsmallest(max(80, n_clouds * 20), "phot_g_mean_mag").copy()

    chosen = anchors.sample(n=n_clouds, random_state=42)[["x", "y", "z"]].to_numpy()

    inner_rows = []
    outer_rows = []

    inner_palette = [
        "rgba(255,140,110,0.06)",
        "rgba(200,140,255,0.05)",
        "rgba(255,100,160,0.05)",
        "rgba(120,220,255,0.045)",
    ]
    outer_palette = [
        "rgba(255,170,130,0.018)",
        "rgba(170,160,255,0.016)",
        "rgba(255,140,190,0.016)",
        "rgba(160,220,255,0.014)",
    ]

    for i, (cx, cy, cz) in enumerate(chosen):
        # compact inner cloud
        inner_n = int(points_per_cloud * 0.45)
        sx1 = rng.uniform(4.0, 8.0)
        sy1 = rng.uniform(4.0, 8.0)
        sz1 = rng.uniform(18.0, 35.0)

        inner_pts = np.column_stack([
            rng.normal(cx, sx1, size=inner_n),
            rng.normal(cy, sy1, size=inner_n),
            rng.normal(cz, sz1, size=inner_n),
        ])
        inner_df = pd.DataFrame(inner_pts, columns=["x", "y", "z"])
        inner_df["size"] = rng.uniform(2.0, 4.0, size=inner_n)
        inner_df["color"] = inner_palette[i % len(inner_palette)]
        inner_rows.append(inner_df)

        # diffuse outer haze
        outer_n = points_per_cloud - inner_n
        sx2 = rng.uniform(8.0, 14.0)
        sy2 = rng.uniform(8.0, 14.0)
        sz2 = rng.uniform(30.0, 55.0)

        outer_pts = np.column_stack([
            rng.normal(cx, sx2, size=outer_n),
            rng.normal(cy, sy2, size=outer_n),
            rng.normal(cz, sz2, size=outer_n),
        ])
        outer_df = pd.DataFrame(outer_pts, columns=["x", "y", "z"])
        outer_df["size"] = rng.uniform(1.0, 2.2, size=outer_n)
        outer_df["color"] = outer_palette[i % len(outer_palette)]
        outer_rows.append(outer_df)

    return (
        pd.concat(inner_rows, ignore_index=True),
        pd.concat(outer_rows, ignore_index=True),
    )


def make_figure(df: pd.DataFrame) -> go.Figure:
    pretty_df = df.nsmallest(2200, "phot_g_mean_mag").copy()
    nebula_inner_df, nebula_outer_df = make_nebula_points(pretty_df)

    dim_df = pretty_df[pretty_df["phot_g_mean_mag"] >= 13.5].copy()
    mid_df = pretty_df[
        (pretty_df["phot_g_mean_mag"] < 13.5) &
        (pretty_df["phot_g_mean_mag"] >= 11.5)
        ].copy()
    bright_df = pretty_df[pretty_df["phot_g_mean_mag"] < 11.5].copy()

    fig = go.Figure()

    # Outer faint haze
    fig.add_trace(
        go.Scatter3d(
            x=nebula_outer_df["x"],
            y=nebula_outer_df["y"],
            z=nebula_outer_df["z"],
            mode="markers",
            marker=dict(
                size=nebula_outer_df["size"],
                color=nebula_outer_df["color"],
                opacity=1.0,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Inner slightly denser mist
    fig.add_trace(
        go.Scatter3d(
            x=nebula_inner_df["x"],
            y=nebula_inner_df["y"],
            z=nebula_inner_df["z"],
            mode="markers",
            marker=dict(
                size=nebula_inner_df["size"],
                color=nebula_inner_df["color"],
                opacity=1.0,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Dim stars
    fig.add_trace(
        go.Scatter3d(
            x=dim_df["x"],
            y=dim_df["y"],
            z=dim_df["z"],
            mode="markers",
            text=dim_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=dim_df["size"] * 0.45,
                color=dim_df["bp_rp_clamped"],
                colorscale="Turbo",
                opacity=0.08,
            ),
            showlegend=False,
        )
    )

    # Mid stars
    fig.add_trace(
        go.Scatter3d(
            x=mid_df["x"],
            y=mid_df["y"],
            z=mid_df["z"],
            mode="markers",
            text=mid_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=mid_df["size"] * 0.8,
                color=mid_df["bp_rp_clamped"],
                colorscale="Turbo",
                opacity=0.38,
            ),
            showlegend=False,
        )
    )

    # Bright stars
    fig.add_trace(
        go.Scatter3d(
            x=bright_df["x"],
            y=bright_df["y"],
            z=bright_df["z"],
            mode="markers",
            text=bright_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=bright_df["size"] * 1.5,
                color=bright_df["bp_rp_clamped"],
                colorscale="Turbo",
                opacity=0.92,
                colorbar=dict(title="BP−RP<br>color", len=0.7),
            ),
            showlegend=False,
        )
    )

    xr = np.nanpercentile(np.abs(pretty_df["x"]), 99)
    yr = np.nanpercentile(np.abs(pretty_df["y"]), 99)
    zr = np.nanpercentile(np.abs(pretty_df["z"]), 99)

    fig.update_layout(
        title=dict(text="Gaia DR3: Orion Region in Local 3D Coordinates", x=0.5),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            bgcolor="black",
            xaxis=dict(
                title="Local X (pc)",
                color="white",
                showgrid=False,
                zeroline=False,
                showbackground=False,
                range=[-xr, xr],
            ),
            yaxis=dict(
                title="Local Y (pc)",
                color="white",
                showgrid=False,
                zeroline=False,
                showbackground=False,
                range=[-yr, yr],
            ),
            zaxis=dict(
                title="Depth (pc)",
                color="white",
                showgrid=False,
                zeroline=False,
                showbackground=False,
                range=[-zr, zr],
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1.5, z=0.45),
            camera=dict(eye=dict(x=1.9, y=1.5, z=0.8)),
        ),
    )

    return fig


def add_camera_orbit_frames(fig: go.Figure, n_frames: int = 90, radius: float = 2.1) -> go.Figure:
    frames = []
    for i in range(n_frames):
        theta = 2 * math.pi * i / n_frames
        eye = dict(
            x=radius * math.cos(theta),
            y=radius * math.sin(theta),
            z=0.8 + 0.25 * math.sin(theta / 2),
        )
        frames.append(go.Frame(layout=dict(scene_camera=dict(eye=eye)), name=f"f{i}"))

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.98,
                buttons=[
                    dict(
                        label="Play orbit",
                        method="animate",
                        args=[None, {"frame": {"duration": 70, "redraw": False}, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ]
    )
    return fig


def main(refresh: bool = False) -> None:
    data = load_dataset(ORION, refresh=refresh)
    data = prepare_orion_data(data)

    fig = make_figure(data)
    fig = add_camera_orbit_frames(fig)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(HTML_FILE, include_plotlyjs="cdn")
    print(f"Saved interactive HTML to {HTML_FILE}")
    fig.show()


if __name__ == "__main__":
    main(refresh=False)
