"""Astronomical coordinate transformations."""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

def parallax_to_distance_pc(parallax_mas: pd.Series) -> pd.Series:
    return 1000.0 / parallax_mas

def add_distance(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["distance_pc"] = parallax_to_distance_pc(result["parallax"])
    return result

def add_local_cartesian(
        data: pd.DataFrame,
        *,
        center_ra_deg: float,
        center_dec_deg: float,
) -> pd.DataFrame:
    """Add local x/y/depth coordinates in parsecs around a sky position."""
    result = add_distance(data)

    center = SkyCoord(
        ra=center_ra_deg * u.deg,
        dec=center_dec_deg * u.deg,
        frame="icrs",
    )

    stars = SkyCoord(
        ra=result["ra"].to_numpy() * u.deg,
        dec=result["dec"].to_numpy() * u.deg,
        distance=result["distance_pc"].to_numpy() * u.pc,
        frame="icrs",
    )

    separation = center.separation(stars).to_value(u.rad)
    position_angle = center.position_angle(stars).to_value(u.rad)
    distance_pc = stars.distance.to_value(u.pc)

    result["x"] = (
            distance_pc
            * np.tan(separation)
            * np.sin(position_angle)
    )
    result["y"] = (
            distance_pc
            * np.tan(separation)
            * np.cos(position_angle)
    )
    result["z"] = distance_pc - np.median(distance_pc)

    return result
