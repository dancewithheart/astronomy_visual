#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astroquery.gaia import Gaia
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


@dataclass(frozen=True)
class QueryConfig:
    ra_deg: float = 83.82208
    dec_deg: float = -5.39111
    radius_deg: float = 3.0
    row_limit: int = 8000
    parallax_min_mas: float = 1.0
    parallax_max_mas: float = 8.0
    parallax_over_error_min: float = 5.0
    gmag_max: float = 15.5


CACHE_DIR = Path("cache")
PARQUET_FILE = CACHE_DIR / "orion_gaia.parquet"
FITS_FILE = CACHE_DIR / "orion_gaia.fits"
HTML_FILE = Path("gaia_orion_local_3d.html")


def build_query(cfg: QueryConfig) -> str:
    return f"""
    SELECT TOP {cfg.row_limit}
        source_id,
        ra,
        dec,
        parallax,
        parallax_over_error,
        phot_g_mean_mag,
        bp_rp,
        random_index
    FROM gaiadr3.gaia_source
    WHERE
        1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {cfg.ra_deg}, {cfg.dec_deg}, {cfg.radius_deg})
        )
        AND parallax IS NOT NULL
        AND parallax BETWEEN {cfg.parallax_min_mas} AND {cfg.parallax_max_mas}
        AND parallax_over_error >= {cfg.parallax_over_error_min}
        AND phot_g_mean_mag IS NOT NULL
        AND bp_rp IS NOT NULL
        AND phot_g_mean_mag < {cfg.gmag_max}
    ORDER BY random_index
    """


def query_gaia_table(cfg: QueryConfig) -> Table:
    Gaia.ROW_LIMIT = -1
    query = build_query(cfg)

    launchers = [
        ("sync", lambda: Gaia.launch_job(query)),
        ("async", lambda: Gaia.launch_job_async(query)),
    ]

    last_error = None
    for name, launcher in launchers:
        try:
            print(f"Trying Gaia query via {name}...")
            time.sleep(1.0)
            job = launcher()
            tbl = job.get_results()
            if len(tbl) == 0:
                raise RuntimeError("Gaia query returned 0 rows")
            print(f"Downloaded {len(tbl):,} rows")
            return tbl
        except Exception as e:
            print(f"{name} query failed: {e}")
            last_error = e

    raise RuntimeError("Both Gaia query methods failed") from last_error


def save_table_local(tbl: Table) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # FITS always works well for astronomy tables
    tbl.write(FITS_FILE, format="fits", overwrite=True)
    print(f"Saved {FITS_FILE}")

    # Parquet is nice for fast reloads if pyarrow is installed
    try:
        tbl.write(PARQUET_FILE, format="parquet", overwrite=True)
        print(f"Saved {PARQUET_FILE}")
    except Exception as e:
        print(f"Parquet save skipped: {e}")


def load_local_table() -> Table | None:
    if PARQUET_FILE.exists():
        print(f"Loading cached Parquet: {PARQUET_FILE}")
        return Table.read(PARQUET_FILE, format="parquet")

    if FITS_FILE.exists():
        print(f"Loading cached FITS: {FITS_FILE}")
        return Table.read(FITS_FILE, format="fits")

    return None


def get_data(cfg: QueryConfig, refresh: bool = False) -> pd.DataFrame:
    if not refresh:
        cached = load_local_table()
        if cached is not None:
            return cached.to_pandas()

    tbl = query_gaia_table(cfg)
    save_table_local(tbl)
    return tbl.to_pandas()


def add_derived_columns(df: pd.DataFrame, center_ra_deg: float, center_dec_deg: float) -> pd.DataFrame:
    df = df.copy()

    df["distance_pc"] = 1000.0 / df["parallax"]
    df["bp_rp_clamped"] = df["bp_rp"].clip(-0.5, 4.0)

    center = SkyCoord(ra=center_ra_deg * u.deg, dec=center_dec_deg * u.deg, frame="icrs")
    stars = SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        distance=df["distance_pc"].to_numpy() * u.pc,
        frame="icrs",
    )

    sep = center.separation(stars)
    pa = center.position_angle(stars)

    dist_pc = stars.distance.to_value(u.pc)

    # Local region coordinates instead of Earth-centered Cartesian
    dx = dist_pc * np.tan(sep.to_value(u.rad)) * np.sin(pa.to_value(u.rad))
    dy = dist_pc * np.tan(sep.to_value(u.rad)) * np.cos(pa.to_value(u.rad))
    dz = dist_pc - np.median(dist_pc)

    df["x"] = dx
    df["y"] = dy
    df["z"] = dz

    brightness = 10 ** (-0.4 * df["phot_g_mean_mag"].to_numpy())
    brightness = brightness / np.nanpercentile(brightness, 99.5)
    df["size"] = 0.35 + 1.5 * np.clip(np.sqrt(brightness), 0, 1.0)

    df["hover"] = (
        "Gaia DR3 source: " + df["source_id"].astype(str)
        + "<br>G mag: " + df["phot_g_mean_mag"].round(2).astype(str)
        + "<br>BP-RP: " + df["bp_rp"].round(2).astype(str)
        + "<br>Distance: " + df["distance_pc"].round(1).astype(str) + " pc"
    )
    return df


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
    cfg = QueryConfig()
    df = get_data(cfg, refresh=refresh)
    df = add_derived_columns(df, cfg.ra_deg, cfg.dec_deg)

    fig = make_figure(df)
    fig = add_camera_orbit_frames(fig)

    fig.write_html(HTML_FILE, include_plotlyjs="cdn")
    print(f"Saved interactive HTML to {HTML_FILE}")
    fig.show()


if __name__ == "__main__":
    main(refresh=False)
