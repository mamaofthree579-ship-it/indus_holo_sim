# simulator/indus_signs.py

INDUS_SIGNS = {
    "NB001": {
        "name": "Fish Sign",
        "default_freq": 22.0,
        "sigma": 0.06,
        "harmonics": [(1, 1, 0), (2, 0.6, 0), (3, 0.3, 0)]
    },
    "NB002": {
        "name": "Jar Sign",
        "default_freq": 16.0,
        "sigma": 0.05,
        "harmonics": [(1, 1, 0)]
    },
    "NB003": {
        "name": "Double Stripes",
        "default_freq": 32.0,
        "sigma": 0.04,
        "harmonics": [(1, 1, 0), (3, 0.4, 0)]
    },
    "NB004": {
        "name": "U-Shape",
        "default_freq": 12.0,
        "sigma": 0.08,
        "harmonics": [(1, 1, 0)]
    },
    "NB005": {
        "name": "Tri-Fork",
        "default_freq": 27.0,
        "sigma": 0.07,
        "harmonics": [(1, 1, 0), (2, 0.5, 0)]
    },
    # Add more here…
}

NB_LIST = list(INDUS_SIGNS.keys())
