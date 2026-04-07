#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
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
    gmag_max: float = 15.5


CACHE_DIR = Path("cache")
PARQUET_FILE = CACHE_DIR / "orion_gaia.parquet"
FITS_FILE = CACHE_DIR / "orion_gaia.fits"
SCREENSHOT_FILE = Path("gaia_orion_pyvista.png")


def build_query(cfg: QueryConfig) -> str:
    return f"""
    SELECT TOP {cfg.row_limit}
        source_id,
        ra,
        dec,
        parallax,
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
        AND phot_g_mean_mag IS NOT NULL
        AND bp_rp IS NOT NULL
        AND phot_g_mean_mag < {cfg.gmag_max}
    ORDER BY random_index
    """


def query_gaia_table(cfg: QueryConfig) -> Table:
    Gaia.ROW_LIMIT = -1
    query = build_query(cfg)
    job = Gaia.launch_job(query)
    tbl = job.get_results()
    if len(tbl) == 0:
        raise RuntimeError("Gaia query returned 0 rows")
    return tbl


def save_table_local(tbl: Table) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tbl.write(FITS_FILE, format="fits", overwrite=True)
    try:
        tbl.write(PARQUET_FILE, format="parquet", overwrite=True)
    except Exception:
        pass


def load_local_table() -> Table | None:
    if PARQUET_FILE.exists():
        return Table.read(PARQUET_FILE, format="parquet")
    if FITS_FILE.exists():
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

    # Local Orion-centered coordinates
    dx = dist_pc * np.tan(sep.to_value(u.rad)) * np.sin(pa.to_value(u.rad))
    dy = dist_pc * np.tan(sep.to_value(u.rad)) * np.cos(pa.to_value(u.rad))
    dz = dist_pc - np.median(dist_pc)

    df["x"] = dx
    df["y"] = dy
    df["z"] = dz

    brightness = 10 ** (-0.4 * df["phot_g_mean_mag"].to_numpy())
    brightness = brightness / np.nanpercentile(brightness, 99.5)
    df["size"] = 3.0 + 10.0 * np.clip(np.sqrt(brightness), 0, 1.0)

    return df


def star_rgb_from_bprp(bp_rp: np.ndarray) -> np.ndarray:
    """
    Approximate star colors from Gaia BP-RP.
    Returns uint8 RGB array.
    """
    x = np.clip(bp_rp, -0.5, 4.0)

    rgb = np.zeros((len(x), 3), dtype=float)

    # piecewise hand-tuned gradient:
    # blue -> white -> yellow -> orange -> red
    for i, v in enumerate(x):
        if v < 0.2:
            t = (v + 0.5) / 0.7
            c0 = np.array([120, 170, 255], dtype=float)
            c1 = np.array([220, 235, 255], dtype=float)
        elif v < 0.8:
            t = (v - 0.2) / 0.6
            c0 = np.array([220, 235, 255], dtype=float)
            c1 = np.array([255, 245, 220], dtype=float)
        elif v < 1.8:
            t = (v - 0.8) / 1.0
            c0 = np.array([255, 245, 220], dtype=float)
            c1 = np.array([255, 210, 120], dtype=float)
        else:
            t = (min(v, 4.0) - 1.8) / 2.2
            c0 = np.array([255, 210, 120], dtype=float)
            c1 = np.array([255, 120, 90], dtype=float)

        rgb[i] = (1 - t) * c0 + t * c1

    return np.clip(rgb, 0, 255).astype(np.uint8)


def build_density_grid(
        df: pd.DataFrame,
        dims: tuple[int, int, int] = (80, 80, 70),
        n_anchor_stars: int = 90,
        z_scale: float = 0.22,
        seed: int = 42,
) -> pv.ImageData:
    rng = np.random.default_rng(seed)

    work = df.copy()
    work["z_render"] = work["z"] * z_scale

    anchors = work.nsmallest(n_anchor_stars, "phot_g_mean_mag").copy()
    anchors = anchors[np.abs(anchors["z_render"]) < np.nanpercentile(np.abs(anchors["z_render"]), 70)]
    if len(anchors) < 20:
        anchors = work.nsmallest(n_anchor_stars, "phot_g_mean_mag").copy()

    x = work["x"].to_numpy()
    y = work["y"].to_numpy()
    z = work["z_render"].to_numpy()

    xr = np.nanpercentile(np.abs(x), 99)
    yr = np.nanpercentile(np.abs(y), 99)
    zr = np.nanpercentile(np.abs(z), 99)

    xmin, xmax = -xr, xr
    ymin, ymax = -yr, yr
    zmin, zmax = -zr, zr

    nx, ny, nz = dims
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    density = np.zeros((nx, ny, nz), dtype=np.float32)

    chosen = anchors.sample(n=min(22, len(anchors)), random_state=seed)

    for _, row in chosen.iterrows():
        cx, cy, cz = row["x"], row["y"], row["z_render"]

        sx = rng.uniform(2.5, 5.5)
        sy = rng.uniform(2.5, 5.5)
        sz = rng.uniform(4.0, 10.0)
        amp = rng.uniform(0.45, 1.0)

        blob = np.exp(
            -(
                    ((X - cx) ** 2) / (2 * sx * sx)
                    + ((Y - cy) ** 2) / (2 * sy * sy)
                    + ((Z - cz) ** 2) / (2 * sz * sz)
            )
        )
        density += amp * blob.astype(np.float32)

    central_blob = np.exp(
        -(
                (X ** 2) / (2 * (0.32 * xr) ** 2)
                + (Y ** 2) / (2 * (0.32 * yr) ** 2)
                + (Z ** 2) / (2 * (0.28 * zr) ** 2)
        )
    )
    secondary_blobs = [
        (-0.18 * xr,  0.10 * yr,  0.05 * zr, 0.10, 0.12, 0.10, 0.16),
        ( 0.22 * xr, -0.08 * yr, -0.02 * zr, 0.09, 0.10, 0.08, 0.13),
        ( 0.05 * xr,  0.20 * yr,  0.00 * zr, 0.08, 0.09, 0.07, 0.11),
    ]
    density += 0.22 * central_blob.astype(np.float32)
    for cx, cy, cz, sxr, syr, szr, amp in secondary_blobs:
        blob = np.exp(
            -(
                    ((X - cx) ** 2) / (2 * (sxr * xr) ** 2)
                    + ((Y - cy) ** 2) / (2 * (syr * yr) ** 2)
                    + ((Z - cz) ** 2) / (2 * (szr * zr) ** 2)
            )
        )
        density += amp * blob.astype(np.float32)
    density /= max(float(density.max()), 1e-6)

    grid = pv.ImageData()
    grid.dimensions = np.array(dims) + 1
    grid.origin = (xmin, ymin, zmin)
    grid.spacing = (
        (xmax - xmin) / nx,
        (ymax - ymin) / ny,
        (zmax - zmin) / nz,
    )
    grid.cell_data["density"] = density.flatten(order="F")
    return grid


def build_star_polydata(df: pd.DataFrame, z_scale: float = 0.22) -> pv.PolyData:
    pts = np.column_stack([
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        df["z"].to_numpy() * z_scale,
        ])
    poly = pv.PolyData(pts)
    poly["size"] = df["size"].to_numpy()
    poly["rgb"] = star_rgb_from_bprp(df["bp_rp_clamped"].to_numpy())
    return poly


def render_scene(df: pd.DataFrame, screenshot: bool = True) -> None:
    z_scale = 0.12
    pretty_df = df.nsmallest(1800, "phot_g_mean_mag").copy()

    grid = build_density_grid(pretty_df, dims=(80, 80, 70), n_anchor_stars=90, z_scale=z_scale)
    stars = build_star_polydata(pretty_df, z_scale=z_scale)

    plotter = pv.Plotter(window_size=(1600, 900))
    plotter.set_background("black")

    opacity = [0.0, 0.0, 0.015, 0.04, 0.08, 0.15, 0.24]
    plotter.add_volume(
        grid,
        scalars="density",
        cmap="magma",
        opacity=opacity,
        shade=True,
        blending="composite",
        show_scalar_bar=False,
    )

    plotter.add_points(
        stars,
        scalars="rgb",
        rgb=True,
        point_size=2.0,
        render_points_as_spheres=True,
        opacity=0.12,
    )

    bright_df = pretty_df[pretty_df["phot_g_mean_mag"] < 11.8].copy()
    bright_stars = build_star_polydata(bright_df, z_scale=z_scale)
    plotter.add_points(
        bright_stars,
        scalars="rgb",
        rgb=True,
        point_size=5.0,
        render_points_as_spheres=True,
        opacity=0.95,
    )

    xr = np.nanpercentile(np.abs(pretty_df["x"]), 99)
    yr = np.nanpercentile(np.abs(pretty_df["y"]), 99)
    zr = np.nanpercentile(np.abs(pretty_df["z"] * z_scale), 99)

    plotter.camera_position = [
        (1.6 * xr, -1.8 * yr, 0.9 * zr),
        (0, 0, 0),
        (0, 0, 1),
    ]

    plotter.add_text(
        "Gaia DR3: Orion Region",
        font_size=10,
        color="white",
    )

    if screenshot:
        plotter.show(screenshot=str(SCREENSHOT_FILE))
        print(f"Saved screenshot to {SCREENSHOT_FILE}")
    else:
        plotter.show()


def main(refresh: bool = False, screenshot: bool = True) -> None:
    cfg = QueryConfig()
    df = get_data(cfg, refresh=refresh)
    df = add_derived_columns(df, cfg.ra_deg, cfg.dec_deg)
    render_scene(df, screenshot=screenshot)


if __name__ == "__main__":
    main(refresh=False, screenshot=False)
