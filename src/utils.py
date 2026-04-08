"""
utils.py

Shared geographic utilities used across pipeline stages.
"""

import numpy as np


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorised Haversine distance (km).

    All arguments may be scalars or numpy arrays; broadcasting applies.
    Returns the same shape as the broadcast of the inputs.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [
        np.asarray(lat1, dtype=float),
        np.asarray(lon1, dtype=float),
        np.asarray(lat2, dtype=float),
        np.asarray(lon2, dtype=float),
    ])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
