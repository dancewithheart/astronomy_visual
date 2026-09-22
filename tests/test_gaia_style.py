import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from gaia import parallax_to_distance_pc

@given(
    st.floats(
        min_value=0.01,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False))
def test_distance_is_positive_for_positive_parallax(parallax: float):
    distance = parallax_to_distance_pc(np.array([parallax]))[0]
    assert distance > 0

@given(
    p1=st.floats(
        min_value=0.01,
        max_value=100,
        allow_nan=False,
        allow_infinity=False),
    p2=st.floats(
        min_value=0.01,
        max_value=100,
        allow_nan=False,
        allow_infinity=False))
def test_larger_parallax_means_smaller_distance(p1: float, p2: float):
    if p1 == p2:
        return
    d1 = parallax_to_distance_pc(np.array([p1]))[0]
    d2 = parallax_to_distance_pc(np.array([p2]))[0]
    if p1 < p2:
        assert d1 > d2
    else:
        assert d2 > d1
