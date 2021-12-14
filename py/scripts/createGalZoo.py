import numpy as np
import pandas as pd


def agreement(probability):
    """Returns measure of agreement as defined in Dieleman et al 2015
    Args:
        probability(array): shape(num_data, num_class)
    """
    n = probability.shape[1]
    H = -1 * np.sum(probability * np.nan_to_num(np.log(probability)), axis=-1)

    return 1 - (H / np.log(n))


# data_path = "/data/bid13/photoZ/data/pasquet2019/"  # Dara
data_path = "/bgfs/jnewman/bid13/photoZ/data/pasquet2019/" #CRC


cat = pd.read_csv(data_path + "galaxyZoo2.csv")

prob = np.array(
    [
        cat["t01_smooth_or_features_a01_smooth_debiased"],
        cat["t01_smooth_or_features_a02_features_or_disk_debiased"],
        cat["t01_smooth_or_features_a03_star_or_artifact_debiased"],
    ]
).T

agreed = agreement(prob) >= 0.8

cat_agreed = cat[agreed]
prob_agreed = prob[agreed]
class_agreed = np.argmax(prob_agreed, axis=-1)
imageID = np.array(cat_agreed["imageID"])


data = np.load(data_path + "sdss_vagc.npz", allow_pickle=True)

mask = class_agreed < 2

cube = data["cube"][imageID][mask]
labels = class_agreed[mask]
specObjID = data["labels"]["specObjID"][imageID][mask]
z = data["labels"]["z"][imageID][mask]
cat = data["labels"][imageID][mask]

np.savez(
    data_path + "sdss_galaxy_zoo_0.8_agreed.npz",
    cube=cube,
    labels=labels,
    specObjID=specObjID,
    z=z,
    cat=cat,
)
