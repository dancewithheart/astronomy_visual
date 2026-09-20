"""astronomy transformations"""

import pandas as pd

def parallax_to_distance_pc(parallax_mas: pd.Series) -> pd.Series:
    return 1000.0 / parallax_mas

def add_distance(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["distance_pc"] = parallax_to_distance_pc(result["parallax"])
    return result
