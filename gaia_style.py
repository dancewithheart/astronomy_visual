from __future__ import annotations

import numpy as np

BP_RP_STOPS = np.array([-0.5, 0.2, 0.8, 1.8, 4.0,])
RGB_STOPS = np.array([
    [120, 170, 255],
    [220, 235, 255],
    [255, 245, 220],
    [255, 210, 120],
    [255, 120, 90],
])

def star_rgb_from_bprp(bp_rp: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(bp_rp, dtype=float), BP_RP_STOPS[0], BP_RP_STOPS[-1])
    channels = [
        np.interp(values, BP_RP_STOPS, RGB_STOPS[:, channel])
        for channel in range(3)
    ]
    return (
        np.column_stack(channels)
        .round()
        .astype(np.uint8)
    )

def marker_size(brightness: np.ndarray, *, minimum: float = 0.4, maximum: float = 2.0) -> np.ndarray:
    brightness = np.clip(brightness, 0, 1)
    return minimum + (maximum - minimum) * np.sqrt(brightness)
