from ast import literal_eval
import json
from pathlib import Path
import pprint

from keras.models import model_from_json
import pandas as pd
import simplejson
import tensorflow as tf
import yaml

from ..base.deepCapsLayers import (
    CapsuleLayer,
    ConvertToCaps,
    FlattenCaps,
    CapsToScalars,
    Conv2DCaps,
    Mask_CID,
    ConvCapsuleLayer3D,
)


def construct_path_out(run_name=None, path=None):
    if (run_name is not None) and (path is None):
        path_results = Path(__file__).parents[2] / "results"
        path = path_results / run_name.split("_")[0] / run_name / "results"
    else:
        path = Path(path).resolve() / run_name.split("_")[0] / run_name / "results"

    return run_name, path


def construct_path_cubes(paths, **params):
    """
    Args:
        paths (Series): File paths to cubes.

    Returns:
        Series: File paths to cubes.
    """
    if not Path(paths.iloc[0]).exists():
        path_cubes = pd.Series([str(Path(params["path_cubes"])) for _ in range(len(paths))])
        cube_ids = paths.str.split("cubes").str[1]
        paths = path_cubes.str.cat(cube_ids)

    return paths


def load_config(config_file, verbose=True):
    """Load configuration file.

    Args:
        config_file (str): Path to the yml config file.
        verbose (bool): If True, print config file to stdout. Default
            is True.

    Returns:
        A dictionary of dictionaries mapping each configuration mode
        to its required set of parameters. The elements are dictionaries
        which map the names of the parameters to their values.

    Raises:
        YAMLError: An error occurred while reading the YAML file
    """
    if not Path(config_file).exists():
        config_file = Path(__file__).parents[1] / "configs" / config_file

    with open(config_file, "r") as stream:
        try:
            all_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    all_config["image_shape"] = literal_eval(all_config["image_shape"])
    if "bands" in all_config.keys():
        all_config["bands"] = tuple([it.strip() for it in all_config["bands"].split(",")])

    for kk, vv in all_config.items():
        if vv == "None":
            all_config[kk] = None

    if all_config["num_gpus"] >= 2:
        all_config["compile_on"] = "cpu"

    if all_config["path_data"] is None:
        all_config["path_data"] = "."

    if verbose:
        print("\nconfig:")
        pprint.PrettyPrinter(indent=4).pprint(all_config)
        print()

    return all_config


def load_model(path_architecture, path_weights, custom_objects=None):
    with open(path_architecture, "r") as fin:
        model_arch = json.load(fin)

    if custom_objects is None:
        custom_objects = {
            "CapsuleLayer": CapsuleLayer,
            "ConvertToCaps": ConvertToCaps,
            "FlattenCaps": FlattenCaps,
            "CapsToScalars": CapsToScalars,
            "Conv2DCaps": Conv2DCaps,
            "Mask_CID": Mask_CID,
            "ConvCapsuleLayer3D": ConvCapsuleLayer3D,
        }

    model_arch_str = json.dumps(model_arch)
    model = model_from_json(model_arch_str, custom_objects=custom_objects)
    model.load_weights(path_weights, by_name=True)
    return model


def model_to_json(model, filename):
    json_string = model.to_json()

    filename_parent = Path(filename).parent
    filename_parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as fout:
        fout.write(simplejson.dumps(simplejson.loads(json_string), indent=4))


def setup_timeline(params):
    params["run_options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    params["run_metadata"] = tf.RunMetadata()
    return params


def write_timeline(params):
    from tensorflow.python.client import timeline

    tl = timeline.Timeline(params["run_metadata"].step_stats)
    trace = tl.generate_chrome_trace_format()
    with open("timeline-{params['run_name'].split('_')[1]:02}.json", "w") as out:
        out.write(trace)
