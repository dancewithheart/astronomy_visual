"""ADQL queries for Gaia online datasets"""
import time

from dataclasses import dataclass

import pandas as pd
from astroquery.gaia import Gaia


DEFAULT_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "bp_rp",
    "ruwe",
)

ACTIVE_PHASES = {
    "PENDING",
    "QUEUED",
    "EXECUTING",
}


@dataclass(frozen=True)
class QueryConfig:
    ra_deg: float
    dec_deg: float
    radius_deg: float

    row_limit: int = 100_000
    table: str = "gaiadr3.gaia_source_lite"

    g_mag_max: float | None = None
    parallax_over_error_min: float | None = None
    parallax_min: float | None = None
    parallax_max: float | None = None
    ruwe_max: float | None = None


def build_query(config: QueryConfig) -> str:
    conditions = [
        (
            "1 = CONTAINS("
            "POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {config.ra_deg}, {config.dec_deg}, "
            f"{config.radius_deg})"
            ")"
        ),
        "parallax IS NOT NULL",
        "pmra IS NOT NULL",
        "pmdec IS NOT NULL",
    ]

    if config.g_mag_max is not None:
        conditions.append(f"phot_g_mean_mag < {config.g_mag_max}")

    if config.ruwe_max is not None:
        conditions.append(f"ruwe < {config.ruwe_max}")

    if config.parallax_over_error_min is not None:
        conditions.append(
            "parallax_over_error >= "
            f"{config.parallax_over_error_min}")
    if config.parallax_min is not None:
        conditions.append(f"parallax >= {config.parallax_min}")
    if config.parallax_max is not None:
        conditions.append(f"parallax <= {config.parallax_max}")

    columns = ",\n    ".join(DEFAULT_COLUMNS)
    where = "\n    AND ".join(conditions)

    return f"""
SELECT TOP {config.row_limit}
    {columns}
FROM {config.table}
WHERE {where}
""".strip()


# def query_gaia(config: QueryConfig) -> pd.DataFrame:
#     job = Gaia.launch_job_async(build_query(config))
#     table = job.get_results()
#     return table.to_pandas()

# # Gaia docs: https://astroquery.readthedocs.io/en/latest/api/astroquery.gaia.GaiaClass.html
def query_gaia(config: QueryConfig) -> pd.DataFrame:
    query = build_query(config)

    print("[Gaia] submitting query...", flush=True)

    started = time.monotonic()

    job = Gaia.launch_job_async(
        query,
        name="astronomy-lab",
        background=True,
        verbose=True,
    )

    print(
        f"[Gaia] job submitted: {job.jobid}",
        flush=True,
    )

    while True:
        phase = job.get_phase(update=True).strip().upper()
        elapsed = time.monotonic() - started

        print(
            f"[Gaia] phase={phase}, elapsed={elapsed:.1f}s",
            flush=True,
        )

        if phase not in ACTIVE_PHASES:
            break

        time.sleep(5)

    if phase != "COMPLETED":
        raise RuntimeError(
            f"Gaia query failed: job={job.jobid}, phase={phase}"
        )

    print("[Gaia] downloading results...", flush=True)

    table = job.get_results()

    print(
        f"[Gaia] received {len(table):,} rows",
        flush=True,
    )

    return table.to_pandas()
