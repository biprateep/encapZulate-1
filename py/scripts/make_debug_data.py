"""
Make single file with all images of galaxies in dummy_data.h5.
"""

from pathlib import Path

import numpy as np
import pandas as pd


path_photoz = Path.home() / "projects" / "photoz"
path_pasquet2019 = path_photoz / "data" / "pasquet2019"

input_shape = (64, 64, 5)

labels = pd.read_hdf(path_pasquet2019 / "sdss_debug.h5")
paths = labels["filePath"]
n_gal = len(paths)

path_cubes = pd.Series([str(path_pasquet2019 / "cubes") for _ in range(n_gal)])
cube_ids = paths.str.split("cubes").str[1]
paths = path_cubes.str.cat(cube_ids)

cube = np.empty((n_gal, *input_shape))
for ii, path in enumerate(paths):
    cube[ii] = np.load(path)

np.savez(path_pasquet2019 / "sdss_debug.npz", labels=labels["z_class"].values, cube=cube)

np.savez(
    path_pasquet2019 / "sdss_unittest.npz",
    labels=labels["z_class"].values[:128],
    cube=cube[:128]
)
