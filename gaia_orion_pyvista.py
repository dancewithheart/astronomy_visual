#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

GIF_FILE = Path("gaia_orion_orbit.gif")
MP4_FILE = Path("gaia_orion_orbit.mp4")

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
    from astroquery.gaia import Gaia
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
        print(f"Loading cached Parquet: {PARQUET_FILE}")
        return Table.read(PARQUET_FILE, format="parquet")
    if FITS_FILE.exists():
        print(f"Loading cached FITS: {FITS_FILE}")
        return Table.read(FITS_FILE, format="fits")
    print("No local cache found")
    return None


def get_data(cfg: QueryConfig, refresh: bool = False) -> pd.DataFrame:
    if not refresh:
        cached = load_local_table()
        if cached is not None:
            print("Using local cached Gaia data")
            return cached.to_pandas()

    print("Querying Gaia archive")
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

def tint_star_rgb_by_position(df: pd.DataFrame, rgb: np.ndarray) -> np.ndarray:
    """
    Subtle warm-left / cool-right tint so stars feel embedded in the nebula.
    Keeps the original Gaia-inspired color, only nudges it slightly.
    """
    out = rgb.astype(np.float32).copy()

    x = df["x_render"].to_numpy()
    y = df["y_render"].to_numpy()

    xr = max(np.nanpercentile(np.abs(x), 99), 1e-6)
    yr = max(np.nanpercentile(np.abs(y), 99), 1e-6)

    # Warm stronger on the left, cool stronger on the right
    warm_w = 1.0 / (1.0 + np.exp((x - 0.02 * xr) / (0.20 * xr)))
    cool_w = 1.0 / (1.0 + np.exp((-x - 0.04 * xr) / (0.20 * xr)))

    # Slight vertical modulation so it does not look like a flat left/right paint job
    warm_w *= 0.90 + 0.15 * np.exp(-((y + 0.08 * yr) ** 2) / (2 * (0.25 * yr) ** 2))
    cool_w *= 0.90 + 0.18 * np.exp(-((y - 0.06 * yr) ** 2) / (2 * (0.26 * yr) ** 2))

    warm_tint = np.array([255, 205, 120], dtype=np.float32)
    cool_tint = np.array([160, 190, 255], dtype=np.float32)

    # Keep this subtle
    warm_alpha = 0.10 * warm_w[:, None]
    cool_alpha = 0.08 * cool_w[:, None]

    out = (1.0 - warm_alpha) * out + warm_alpha * warm_tint
    out = (1.0 - cool_alpha) * out + cool_alpha * cool_tint

    return np.clip(out, 0, 255).astype(np.uint8)

def build_density_grid(
        df: pd.DataFrame,
        dims: tuple[int, int, int] = (112, 112, 80),
        n_anchor_stars: int = 90,
        seed: int = 42,
) -> tuple[pv.ImageData, pv.ImageData]:
    rng = np.random.default_rng(seed)

    anchors = df.nsmallest(n_anchor_stars, "phot_g_mean_mag").copy()
    anchors = anchors[np.abs(anchors["z_render"]) < np.nanpercentile(np.abs(anchors["z_render"]), 70)]
    if len(anchors) < 20:
        anchors = df.nsmallest(n_anchor_stars, "phot_g_mean_mag").copy()

    x = df["x_render"].to_numpy()
    y = df["y_render"].to_numpy()
    z = df["z_render"].to_numpy()

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

    base_density = np.zeros((nx, ny, nz), dtype=np.float32)

    chosen = anchors.sample(n=min(22, len(anchors)), random_state=seed)

    for _, row in chosen.iterrows():
        cx, cy, cz = row["x_render"], row["y_render"], row["z_render"]

        sx = rng.uniform(3.0, 6.2)
        sy = rng.uniform(3.0, 6.2)
        sz = rng.uniform(3.0, 7.5)
        amp = rng.uniform(0.42, 0.95)

        blob = np.exp(
            -(
                    ((X - cx) ** 2) / (2 * sx * sx)
                    + ((Y - cy) ** 2) / (2 * sy * sy)
                    + ((Z - cz) ** 2) / (2 * sz * sz)
            )
        )
        base_density += amp * blob.astype(np.float32)

    central_blob = np.exp(
        -(
                (X ** 2) / (2 * (0.30 * xr) ** 2)
                + (Y ** 2) / (2 * (0.30 * yr) ** 2)
                + (Z ** 2) / (2 * (0.24 * zr) ** 2)
        )
    )
    base_density += 0.07 * central_blob.astype(np.float32)

    secondary_blobs = [
        (-0.22 * xr,  0.04 * yr,  0.02 * zr, 0.12, 0.10, 0.10, 0.15),
        ( 0.24 * xr, -0.08 * yr, -0.02 * zr, 0.10, 0.10, 0.09, 0.13),
        ( 0.03 * xr,  0.20 * yr,  0.00 * zr, 0.08, 0.09, 0.08, 0.10),
    ]

    for cx, cy, cz, sxr, syr, szr, amp in secondary_blobs:
        blob = np.exp(
            -(
                    ((X - cx) ** 2) / (2 * (sxr * xr) ** 2)
                    + ((Y - cy) ** 2) / (2 * (syr * yr) ** 2)
                    + ((Z - cz) ** 2) / (2 * (szr * zr) ** 2)
            )
        )
        base_density += amp * blob.astype(np.float32)

    # Mild internal turbulence to break smooth banding
    noise = (
            0.022 * np.sin(0.23 * X + 0.11 * Y)
            + 0.016 * np.sin(0.17 * Y - 0.19 * Z)
            + 0.012 * np.sin(0.21 * X + 0.14 * Z)
    )
    base_density *= (1.0 + noise.astype(np.float32))
    base_density = np.clip(base_density, 0.0, None)

    base_density /= max(float(base_density.max()), 1e-6)

    # Left-right split for color asymmetry.
    # warm strongest on the left, cool strongest on the right
    sigmoid_scale = 0.22 * xr
    warm_weight = 1.0 / (1.0 + np.exp((X - 0.02 * xr) / sigmoid_scale))
    cool_weight = 1.0 / (1.0 + np.exp((-X - 0.04 * xr) / sigmoid_scale))

    # small vertical modulation so it doesn't look like a simple left/right paint job
    warm_mod = 0.85 + 0.25 * np.exp(-((Y + 0.10 * yr) ** 2) / (2 * (0.22 * yr) ** 2))
    cool_mod = 0.80 + 0.30 * np.exp(-((Y - 0.08 * yr) ** 2) / (2 * (0.24 * yr) ** 2))

    warm_density = base_density * warm_weight * warm_mod
    cool_density = base_density * cool_weight * cool_mod

    warm_density /= max(float(warm_density.max()), 1e-6)
    cool_density /= max(float(cool_density.max()), 1e-6)

    def make_grid(arr: np.ndarray) -> pv.ImageData:
        grid = pv.ImageData()
        grid.dimensions = np.array(dims) + 1
        grid.origin = (xmin, ymin, zmin)
        grid.spacing = (
            (xmax - xmin) / nx,
            (ymax - ymin) / ny,
            (zmax - zmin) / nz,
        )
        grid.cell_data["density"] = arr.flatten(order="F")
        return grid

    return make_grid(warm_density), make_grid(cool_density)


def build_star_polydata(df: pd.DataFrame) -> pv.PolyData:
    pts = np.column_stack([
        df["x_render"].to_numpy(),
        df["y_render"].to_numpy(),
        df["z_render"].to_numpy(),
    ])
    poly = pv.PolyData(pts)
    poly["size"] = df["size"].to_numpy()

    rgb = star_rgb_from_bprp(df["bp_rp_clamped"].to_numpy())
    rgb = tint_star_rgb_by_position(df, rgb)
    poly["rgb"] = rgb

    return poly


def render_scene(
        df: pd.DataFrame,
        screenshot: bool = True,
        animate: bool = False,
        movie_format: str = "mp4",
        n_frames: int = 120,
) -> None:
    # pretty_df = df.nsmallest(1200, "phot_g_mean_mag").copy()
    # pretty_df = add_render_columns(pretty_df, x_scale=1.18, y_scale=1.18, z_scale=0.16)

    # fewer stars = cleaner subject
    pretty_df = df.nsmallest(1000, "phot_g_mean_mag").copy()
    # ider x/y + shorter z = less plume-like, more nebula-like
    pretty_df = add_render_columns(pretty_df, x_scale=1.28, y_scale=1.24, z_scale=0.13)

    warm_grid, cool_grid = build_density_grid(pretty_df, dims=(112, 112, 80), n_anchor_stars=90)
    stars = build_star_polydata(pretty_df)

    plotter = pv.Plotter(window_size=(1600, 912), off_screen=(screenshot or animate))
    plotter.set_background("black")

    warm_opacity = [0.0, 0.0, 0.008, 0.022, 0.050, 0.090, 0.145]
    cool_opacity = [0.0, 0.0, 0.006, 0.018, 0.040, 0.075, 0.115]

    plotter.add_volume(
        warm_grid,
        scalars="density",
        cmap="autumn",
        opacity=warm_opacity,
        shade=True,
        blending="composite",
        show_scalar_bar=False,
    )

    plotter.add_volume(
        cool_grid,
        scalars="density",
        cmap="BuPu",
        opacity=cool_opacity,
        shade=True,
        blending="composite",
        show_scalar_bar=False,
    )

    # Split stars by rendered depth for better layering
    zq1 = pretty_df["z_render"].quantile(0.33)
    zq2 = pretty_df["z_render"].quantile(0.66)

    background_df = pretty_df[pretty_df["z_render"] > zq2].copy()
    embedded_df = pretty_df[
        (pretty_df["z_render"] >= zq1) & (pretty_df["z_render"] <= zq2)
        ].copy()
    foreground_df = pretty_df[pretty_df["z_render"] < zq1].copy()

    background_stars = build_star_polydata(background_df)
    embedded_stars = build_star_polydata(embedded_df)
    foreground_stars = build_star_polydata(foreground_df)

    # Background stars: faint and tiny
    plotter.add_points(
        background_stars,
        scalars="rgb",
        rgb=True,
        point_size=1.2,
        render_points_as_spheres=True,
        opacity=0.035,
    )

    # Embedded stars: normal
    plotter.add_points(
        embedded_stars,
        scalars="rgb",
        rgb=True,
        point_size=1.9,
        render_points_as_spheres=True,
        opacity=0.09,
    )

    # Foreground stars: a bit larger
    plotter.add_points(
        foreground_stars,
        scalars="rgb",
        rgb=True,
        point_size=2.8,
        render_points_as_spheres=True,
        opacity=0.16,
    )

    # Bright stars on top
    bright_df = pretty_df[pretty_df["phot_g_mean_mag"] < 11.8].copy()
    bright_stars = build_star_polydata(bright_df)
    plotter.add_points(
        bright_stars,
        scalars="rgb",
        rgb=True,
        point_size=4.8,
        render_points_as_spheres=True,
        opacity=0.95,
    )

    xr = np.nanpercentile(np.abs(pretty_df["x_render"]), 99)
    yr = np.nanpercentile(np.abs(pretty_df["y_render"]), 99)
    zr = np.nanpercentile(np.abs(pretty_df["z_render"]), 99)

    focal = np.array([0.06 * xr, -0.03 * yr, 0.0])

    plotter.camera_position = [
        (1.45 * xr, -1.75 * yr, 0.85 * zr),
        tuple(focal),
        (0, 0, 1),
    ]

    if not animate:
        plotter.add_text(
            "Gaia DR3: Orion Region",
            font_size=10,
            color="white",
        )

    if animate:
        if movie_format == "gif":
            plotter.open_gif(str(GIF_FILE), fps=18)
            out_name = GIF_FILE
        else:
            plotter.open_movie(str(MP4_FILE), framerate=24)
            out_name = MP4_FILE

        # Must show with auto_close=False before orbiting on a path
        plotter.show(auto_close=False)

        # custom shallow tilted elliptical orbit
        angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

        # tighter orbit
        # subject stays bigger in frame
        # feels less like surveying data, more like circling an object
        radius_x = 1.10 * xr
        radius_y = 1.22 * yr
        base_z = 0.52 * zr

        for theta in angles:
            cam = (
                radius_x * np.cos(theta),
                radius_y * np.sin(theta),
                base_z + 0.18 * zr * np.sin(theta + 0.6),
            )
            plotter.camera_position = [cam, tuple(focal), (0, 0, 1)]
            plotter.write_frame()

        plotter.close()
        print(f"Saved orbit animation to {out_name}")
        return

    if screenshot:
        plotter.show(screenshot=str(SCREENSHOT_FILE))
        print(f"Saved screenshot to {SCREENSHOT_FILE}")
    else:
        plotter.show()

def add_render_columns(
        df: pd.DataFrame,
        x_scale: float = 1.28,
        y_scale: float = 1.24,
        z_scale: float = 0.13,
) -> pd.DataFrame:
    out = df.copy()
    out["x_render"] = out["x"] * x_scale
    out["y_render"] = out["y"] * y_scale
    out["z_render"] = out["z"] * z_scale
    return out

def main(
        refresh: bool = False,
        screenshot: bool = True,
        animate: bool = False,
        movie_format: str = "mp4",
        n_frames: int = 120,
) -> None:
    cfg = QueryConfig()
    df = get_data(cfg, refresh=refresh)
    df = add_derived_columns(df, cfg.ra_deg, cfg.dec_deg)
    render_scene(
        df,
        screenshot=screenshot,
        animate=animate,
        movie_format=movie_format,
        n_frames=n_frames,
    )


if __name__ == "__main__":
    main(refresh=False, screenshot=False, animate=True, movie_format="mp4", n_frames=120)

# if __name__ == "__main__":
#     main(refresh=False, screenshot=False, animate=True, movie_format="gif", n_frames=72)
