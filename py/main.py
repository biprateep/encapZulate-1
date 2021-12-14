import argparse
from base.loss import central_bias
import copy
import time
from shutil import copyfile

import tensorflow as tf
import keras
from keras import optimizers
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capsule Network for estimating photometric redshifts."
    )
    parser.add_argument(
        "config",
        help="Name of config file (checks current directory first then looks "
        "in 'configs' directory).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="train",
        help="Available modes: 'train', 'evaluate', 'predict'.",
    )

    args = parser.parse_args()

    assert args.mode in [
        "train",
        "evaluate",
        "predict",
    ], f"'{args.mode}' is not one of the available modes: 'train', 'evaluate', 'predict'."

    return args


def main():
    print("Tensorflow version:" + str(tf.__version__))
    print("Keras version:" + str(keras.__version__))

    # TODO figure out how to do this with TF2

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    args = parse_args()
    config_ = load_config(args.config)
    config = copy.deepcopy(config_)
    __, config["path_results"] = construct_path_out(
        config["run_name"], config["path_results"]
    )

    # TODO sort out parallelising with tf.strategy
    if config["dataset"] in ["mnist", "cifar10", "cifar100"]:
        (x_train, y_train), (x_test, y_test) = load_data(**config)

    else:
        (
            (x_train, y_train, vals_train, z_spec_train),
            (x_dev, y_dev, vals_dev, z_spec_dev),
            (_, _, _, _),
        ) = load_data(**config)

    config["input_shape"] = x_train.shape[1:]

    # import the appropriate model
    CapsNet = import_model(model_name=config["model_name"])

    kwargs = {**config}

    if config["timeline"]:
        kwargs = setup_timeline(kwargs)

    if config["compile_on"] == "cpu":
        with tf.device("/cpu:0"):
            (
                train_model,
                eval_model,
                manipulate_model,
                decoder_model,
                redshift_model,
            ) = CapsNet(**kwargs)
    else:
        (
            train_model,
            eval_model,
            manipulate_model,
            decoder_model,
            redshift_model,
        ) = CapsNet(**kwargs)

    try:
        parallel_train_model = MultiGPUModel(train_model, gpus=config["num_gpus"])
        parallel_eval_model = MultiGPUModel(eval_model, gpus=config["num_gpus"])
        parallel_manipulate_model = MultiGPUModel(
            manipulate_model, gpus=config["num_gpus"]
        )
        parallel_decoder_model = MultiGPUModel(decoder_model, gpus=config["num_gpus"])
        parallel_redshift_model = MultiGPUModel(redshift_model, gpus=config["num_gpus"])
    except ValueError:
        parallel_train_model = train_model
        parallel_eval_model = eval_model
        parallel_manipulate_model = manipulate_model
        parallel_decoder_model = decoder_model
        parallel_redshift_model = redshift_model

    if config["dataset"] in ["mnist", "cifar10", "cifar100"]:
        compile_kwargs = {
            "optimizer": optimizers.Adam(lr=config["learning_rate"]),
            "loss": [margin_loss, "mse"],
            "loss_weights": [1.0, config["lam_recon"] * np.prod(config["input_shape"])],
            "metrics": {"capsnet": "accuracy"},
        }

    else:
        if config["num_quantiles"]:
            print("Using Quantile loss.")
            compile_kwargs = {
                "optimizer": optimizers.Adam(lr=config["learning_rate"]),
                "loss": [
                    margin_loss,
                    "mse",
                    quantile_loss(num_quantiles=config["num_quantiles"]),
                ],
                "loss_weights": [
                    1.0,
                    config["lam_recon"] * np.prod(config["input_shape"]),
                    config["lam_redshift"],
                ],
                "metrics": {
                    "capsnet": "accuracy",
                    "redshift_model": central_mse(**config),
                },
            }
        else:
            print("Using MSE Loss.")
            compile_kwargs = {
                "optimizer": optimizers.Adam(lr=config["learning_rate"]),
                "loss": [margin_loss, "mse", "mse"],
                "loss_weights": [
                    1.0,
                    config["lam_recon"] * np.prod(config["input_shape"]),
                    config["lam_redshift"],
                ],
                "metrics": {
                    "capsnet": "accuracy",
                    "redshift_model": [central_mse(**config), central_bias(**config)],
                },
            }

    parallel_train_model.compile(**compile_kwargs)
    # parallel_eval_model.compile(**compile_kwargs)
    # TODO Should manipulate_model and decoder be compiled
    # parallel_manipulate_model.compile(**compile_kwargs)

    train_model.summary()

    model_to_json(
        model=train_model, filename=config["path_results"] / "train_model.json"
    )
    model_to_json(model=eval_model, filename=config["path_results"] / "eval_model.json")
    model_to_json(
        model=manipulate_model,
        filename=config["path_results"] / "manipulate_model.json",
    )
    model_to_json(
        model=decoder_model,
        filename=config["path_results"] / "decoder_model.json",
    )
    model_to_json(
        model=redshift_model,
        filename=config["path_results"] / "redshift_model.json",
    )

    if args.mode == "train":
        # Keep a copy of the config file with the results for reference
        copyfile(args.config, str(config["path_results"] / "config.yml"))

        start = time.time()
        parallel_train_model = train(
            model=parallel_train_model,
            data=(
                (x_train, y_train, vals_train, z_spec_train),
                (x_dev, y_dev, vals_dev, z_spec_dev),
            ),
            **config,
        )
        print(f"\nTraining took {time.time() - start:02} seconds\n")

    elif args.mode == "evaluate":
        assert config["eval_on"] in [
            "train",
            "test",
        ], "Select from either the train or test set."
        if config["eval_on"] == "train":
            evaluate(
                model=parallel_eval_model,
                data=(x_train, y_train, z_spec_train),
                **config,
            )
        else:
            evaluate(
                model=parallel_eval_model, data=(x_dev, y_dev, z_spec_dev), **config
            )

    elif args.mode == "predict":
        # TODO allow new data to be input
        y_classes, y_prob, recon = predict(
            model=parallel_eval_model, data=x_dev, **config
        )
        print(y_prob.shape)
    if config["timeline"]:
        write_timeline(kwargs)


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from encapzulate.base.loss import (
        margin_loss,
        quantile_loss,
        central_mse,
        central_bias,
        mqe,
    )
    from encapzulate.base.run_model import evaluate, predict, train
    from encapzulate.data_loader.data_loader import load_data
    from encapzulate.models.multi_gpu import MultiGPUModel
    from encapzulate.utils.fileio import (
        construct_path_out,
        load_config,
        model_to_json,
        setup_timeline,
        write_timeline,
    )
    from encapzulate.utils.utils import git_commit_hash, import_model

    print(f"\n\ngit commit hash: {git_commit_hash(levels=1)}")

    main()
