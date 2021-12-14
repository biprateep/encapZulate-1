from pathlib import Path

import numpy as np
import pandas as pd

if "ihome" in str(Path.home()):
    path_photoz = Path.home() / "photoz"
    # path_photoz = Path("/bgfs") / "jnewman" / "bid13" / "photoZ"
elif "/Users/andrews" in str(Path.home()):
    path_photoz = Path.home() / "projects" / "photoz"

path_pasquet2019 = path_photoz / "data" / "pasquet2019"
path_out = path_pasquet2019 / "batches" / "dummy_data"
path_out.mkdir(exist_ok=True)

batch_size = 32
input_shape = (64, 64, 5)

labels = pd.read_hdf(path_pasquet2019 / "dummy_data.h5")
paths = labels["filePath"].iloc[:batch_size]

batches_per_epoch = int(np.floor(len(labels) / batch_size))

path_cubes = pd.Series([str(path_pasquet2019 / "cubes") for _ in range(len(paths))])
cube_ids = paths.str.split("cubes").str[1]
paths = path_cubes.str.cat(cube_ids)

for ii in range(batches_per_epoch):

    out = np.empty((batch_size, *input_shape))
    ind_start = batches_per_epoch * batch_size
    ind_end = ind_start + batch_size
    for jj, path in enumerate(paths[ind_start:ind_end]):
        out[jj] = np.load(path)

    np.save(path_out / f"batch-{ii:05}.npy", out)
