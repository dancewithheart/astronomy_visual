import numpy as np
import pandas as pd
import pytest

from gaia import (
    QueryConfig,
    add_derived_columns,
    build_query,
    clean_gaia_data,
)

def sample_data() -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": [1, 2, 3],
        "ra": [83.82, 83.90, 84.00],
        "dec": [-5.39, -5.30, -5.20],
        "parallax": [2.0, 4.0, 0.2],
        "parallax_over_error": [10.0, 8.0, 2.0],
        "phot_g_mean_mag": [10.0, 12.0, 14.0],
        "bp_rp": [0.0, 1.0, 2.0],
    })

def test_build_query_uses_configuration():
    config = QueryConfig(row_limit=1234, radius_deg=2.5)
    query = build_query(config)
    assert "TOP 1234" in query
    assert "2.5" in query

def test_clean_data_removes_low_quality_rows():
    config = QueryConfig()
    result = clean_gaia_data(sample_data(), config)
    assert list(result["source_id"]) == [1, 2]

def test_add_derived_columns_calculates_distance():
    config = QueryConfig()
    clean = clean_gaia_data(sample_data(), config)
    result = add_derived_columns(clean, config)
    assert result["distance_pc"].to_list() == pytest.approx([500.0, 250.0])

def test_depth_is_centered_around_median():
    config = QueryConfig()
    clean = clean_gaia_data(sample_data(), config)
    result = add_derived_columns(clean, config)
    assert np.median(result["z"]) == pytest.approx(0)

def test_input_dataframe_is_not_modified():
    config = QueryConfig()
    original = sample_data()
    before = original.copy(deep=True)
    clean_gaia_data(original, config)
    pd.testing.assert_frame_equal(original, before)