from pathlib import Path

import keras
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import to_categorical  # Does One-hot-encoding
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils.data import consolidate_bins, crop_center


def load_data(load_cat=False, **config):
    if config["dataset"] == "mnist":
        (x_train, y_train), (x_test, y_test) = load_mnist(**config)

    elif config["dataset"] == "cifar10":
        (x_train, y_train), (x_test, y_test) = load_cifar10(**config)

    elif config["dataset"] == "cifar100":
        (x_train, y_train), (x_test, y_test) = load_cifar100(**config)

    elif "sdss" in config["dataset"]:
        if load_cat:

            (
                (x_train, y_train, vals_train, z_spec_train, cat_train),
                (x_dev, y_dev, vals_dev, z_spec_dev, cat_dev),
                (x_test, y_test, vals_test, z_spec_test, cat_test),
            ) = load_sdss(load_cat=load_cat, **config)
            return (
                (x_train, y_train, vals_train, z_spec_train, cat_train),
                (x_dev, y_dev, vals_dev, z_spec_dev, cat_dev),
                (x_test, y_test, vals_test, z_spec_test, cat_test),
            )
        else:

            (
                (x_train, y_train, vals_train, z_spec_train),
                (x_dev, y_dev, vals_dev, z_spec_dev),
                (x_test, y_test, vals_test, z_spec_test),
            ) = load_sdss(load_cat=load_cat, **config)
            return (
                (x_train, y_train, vals_train, z_spec_train),
                (x_dev, y_dev, vals_dev, z_spec_dev),
                (x_test, y_test, vals_test, z_spec_test),
            )

    else:
        raise ValueError(
            f"`{config['dataset']}` is not one of the valid datasets: "
            "'mnist', 'cifar10', and 'sdss'."
        )

    return (x_train, y_train), (x_test, y_test)


def load_mnist(num_class, **params):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train.astype("float32"), num_class)
    y_test = to_categorical(y_test.astype("float32"), num_class)
    return (x_train, y_train), (x_test, y_test)


def load_cifar10(num_class, **params):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)
    return (x_train, y_train), (x_test, y_test)


def load_cifar100(num_class, **params):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)
    return (x_train, y_train), (x_test, y_test)


def load_sdss(
    num_class,
    path_data,
    frac_train=0.8,
    frac_dev=0.1,
    random_state=200,
    image_scale=10.0,
    load_cat=False,
    **params,
):
    filename = f"{params['dataset']}.npz"
    paths = [
        Path(path_data),
        Path("/bgfs/jnewman/bid13/photoZ/data/pasquet2019"),
        Path("/Users/andrews/projects/photoz/data/pasquet2019"),
        Path("/home/biprateep/Documents/photozCapsNet/photozCapsNet"),
    ]

    data = None
    for path in paths:
        try:
            data = np.load(str(path / filename), allow_pickle=True)
            break
        except FileNotFoundError:
            continue

    if data is None:
        raise FileNotFoundError

    n_gal = len(data["labels"])
    np.random.seed(random_state)
    indices = np.random.permutation(n_gal)
    ind_split_train = int(np.ceil(frac_train * n_gal))
    ind_split_dev = ind_split_train + int(np.ceil(frac_dev * n_gal))

    # ind_bands = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
    # bands = params.get("bands", ("u", "g", "r", "i", "z"))
    # channels = np.array([ind_bands[band] for band, ind_band in zip(bands, ind_bands)])
    # slice_y, slice_x = crop_center(data["cube"].shape[1:3], params["image_shape"])
    # images = data["cube"][:, slice(*slice_y), slice(*slice_x), channels]
    # labels = consolidate_bins(data["labels"], n_bins_in=num_class, n_bins_out=num_class)
    images = data["cube"]
    labels = data["labels"]
    labels = keras.utils.to_categorical(labels, num_classes=num_class)
    z_spec = data["z"]
    cat = data["cat"]
    vals = pd.DataFrame()
    vals["u-g"] = (cat["modelMag_u"] - cat["extinction_u"]) - (
        cat["modelMag_g"] - cat["extinction_g"]
    )
    vals["g-r"] = (cat["modelMag_g"] - cat["extinction_g"]) - (
        cat["modelMag_r"] - cat["extinction_r"]
    )
    vals["r-i"] = (cat["modelMag_r"] - cat["extinction_r"]) - (
        cat["modelMag_i"] - cat["extinction_i"]
    )
    vals["i-z"] = (cat["modelMag_i"] - cat["extinction_i"]) - (
        cat["modelMag_z"] - cat["extinction_z"]
    )
    vals["EBV"] = cat["EBV"]
    vals["r"] = cat["cModelMag_r"] - cat["extinction_r"]

    scaler = StandardScaler()
    vals = scaler.fit_transform(np.array(vals))

    if params["logistic"]:
        z_spec = np.log((z_spec - params["z_min"]) / (params["z_max"] - z_spec))

    x_train = images[indices[:ind_split_train]] / float(image_scale)
    x_dev = images[indices[ind_split_train:ind_split_dev]] / float(image_scale)
    x_test = images[indices[ind_split_dev:]] / float(image_scale)

    y_train = labels[indices[:ind_split_train]]
    y_dev = labels[indices[ind_split_train:ind_split_dev]]
    y_test = labels[indices[ind_split_dev:]]

    z_spec_train = z_spec[indices[:ind_split_train]]
    z_spec_dev = z_spec[indices[ind_split_train:ind_split_dev]]
    z_spec_test = z_spec[indices[ind_split_dev:]]

    vals_train = vals[indices[:ind_split_train]]
    vals_dev = vals[indices[ind_split_train:ind_split_dev]]
    vals_test = vals[indices[ind_split_dev:]]

    if load_cat == False:
        return (
            (x_train, y_train, vals_train, z_spec_train),
            (x_dev, y_dev, vals_dev, z_spec_dev),
            (x_test, y_test, vals_test, z_spec_test),
        )
    if load_cat == True:
        cat_train = cat[indices[:ind_split_train]]
        cat_dev = cat[indices[ind_split_train:ind_split_dev]]
        cat_test = cat[indices[ind_split_dev:]]
        return (
            (x_train, y_train, vals_train, z_spec_train, cat_train),
            (x_dev, y_dev, vals_dev, z_spec_dev, cat_dev),
            (x_test, y_test, vals_test, z_spec_test, cat_test),
        )
