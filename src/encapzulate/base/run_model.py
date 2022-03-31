import keras
from keras.callbacks import (
    CSVLogger,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

from ..utils.metrics import Metrics, probs_to_redshifts


class DataGenerator(keras.utils.Sequence):
    "Generates custom batches. "

    def __init__(
        self,
        x_train,
        y_train,
        vals_train,
        z_spec,
        batch_size,
        img_augmentation=1,
        use_vals=False,
    ):
        "Initialization"
        self.x_train = x_train
        self.y_train = y_train
        self.z_spec = z_spec
        self.vals_train = vals_train
        self.batch_size = batch_size
        self.epoch = 0
        self.img_augmentation = int(img_augmentation)
        self.use_vals = use_vals
        self.on_epoch_end()

    def __len__(self):
        "calculates the number of batches per epoch"
        if self.img_augmentation:
            return int(
                np.ceil(self.img_augmentation * len(self.y_train) / self.batch_size)
            )
        else:
            return int(np.ceil(len(self.y_train) / self.batch_size))

    def __getitem__(self, idx):
        "Generate one batch of data"
        # Generate indexes of the batch
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]  # uniform batching

        # Generate data
        batch_x = self.x_train[batch_indices]
        batch_y = self.y_train[batch_indices]
        batch_z = self.z_spec[batch_indices]
        batch_vals = self.vals_train[batch_indices]

        if self.img_augmentation:
            batch_x = self.__image_augmentation(batch_x)
        if self.use_vals:
            return [batch_x, batch_y, batch_vals], [batch_y, batch_x, batch_z]
        else:
            return [batch_x, batch_y], [batch_y, batch_x, batch_z]

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.epoch += 1
        if self.img_augmentation:
            self.indices = np.random.permutation(
                self.img_augmentation * len(self.y_train)
            ) % len(self.y_train)
        else:
            self.indices = np.random.permutation(len(self.y_train))

    def __image_augmentation(self, imgs):
        transformations = [1, 0]
        # 1 is rotate, 0 is flip
        img_transform = np.random.choice(
            transformations, size=len(imgs), p=[0.67, 0.33]
        )
        for i in range(len(imgs)):
            if img_transform[i]:
                k = np.random.choice([0, 1, 2, 3])
                imgs[i] = np.rot90(imgs[i], k)
            else:
                k = np.random.choice([np.fliplr, np.flipud])
                imgs[i] = k(imgs[i])
        return imgs


def train(
    model,
    data=None,
    training_generator=None,
    validation_generator=None,
    input_shape=None,
    batch_size=None,
    learning_rate=None,
    decay_rate=None,
    epochs=None,
    checkpoint=None,
    lam_recon=None,
    **params,
):
    """Train a CapsuleNet.

    Args:
        model: The CapsuleNet model.
        data (tuple): Tuple containing training and testing data, like
            `((x_train, y_train), (x_test, y_test))`. Default is None.
        training_generator: Training data generator. Default is None.
        validation_generator: Validation data generator.  Default is
            None.
        input_shape (tuple): Shape of input data.  Default is None.
        batch_size (int): Batch size. Default is None.
        learing_rate (float): Learning rate. Default is None.
        decay_rate (float): Learning rate decay rate . Default is None.
        epochs (int): Number of epochs. Default is None.
        checkpoint (int): Epoch at which to start training. Default is
            None.
        lam_recon (float): Constant that determines the strength of
            regularization by reconstruction loss.

    Returns:
        Trained model.
    """
    path_logs = params["path_results"] / "logs"
    path_weights = params["path_results"] / "weights"
    path_logs.mkdir(parents=True, exist_ok=True)
    path_weights.mkdir(parents=True, exist_ok=True)

    # callbacks
    log = CSVLogger(str(path_logs / "log.csv"))
    tb = TensorBoard(
        log_dir=str(path_logs / "tensorboard-logs"),
        batch_size=batch_size,
        histogram_freq=0,
    )

    # TODO Need better place to keep them
    def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = np.floor(n_epochs / n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max / 2 * (np.cos(cos_inner) + 1)

    lr_decay = LearningRateScheduler(
        schedule=lambda epoch: learning_rate * (decay_rate ** epoch)
    )
    # lr_decay = LearningRateScheduler(
    #     schedule=lambda epoch: cosine_annealing(epoch, epochs, 5, learning_rate)
    # )
    cp = ModelCheckpoint(
        filepath=str(path_weights / "weights-{epoch:02d}.h5"),
        monitor="val_capsnet_acc",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
        mode="max",
    )

    if checkpoint is not None:
        initial_epoch = checkpoint
        model.load_weights(
            params["path_results"] / "weights" / f"weights-{checkpoint:02}.h5"
        )
    else:
        initial_epoch = 0

    (x_train, y_train, vals_train, z_spec_train), (
        x_dev,
        y_dev,
        vals_dev,
        z_spec_dev,
    ) = data

    training_generator = DataGenerator(
        x_train,
        y_train,
        vals_train,
        z_spec_train,
        batch_size,
        img_augmentation=params["img_augmentation"],
        use_vals=params["use_vals"],
    )
    validation_generator = DataGenerator(
        x_dev,
        y_dev,
        vals_dev,
        z_spec_dev,
        batch_size,
        img_augmentation=0,
        use_vals=params["use_vals"],
    )

    if params["img_augmentation"]:
        model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            epochs=epochs,
            initial_epoch=0,
            callbacks=[log, tb, cp, lr_decay],
            use_multiprocessing=True,
            workers=12,
        )
    else:
        model.fit(
            [x_train, y_train, z_spec_train],
            [y_train, x_train, z_spec_train],
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=[[x_dev, y_dev, z_spec_dev], [y_dev, x_dev, z_spec_dev]],
            callbacks=[log, tb, cp, lr_decay],
        )

    return model


def evaluate(model, data, checkpoint_eval, **params):
    """Evaluate a CapsuleNet.

    Args:
        model: The CapsuleNet model.
        data: Testing data generator.
        checkpoint (int): Checkpoint number.

    Returns:
        list: loss, capsnet_loss, decoder_loss, capsnet_acc
    """
    x_test, y_test, z_spec = data
    assert (
        type(checkpoint_eval) == int
    ), "Provide a checkpoint in the config file to load model"

    model.load_weights(
        params["path_results"] / "weights" / f"weights-{checkpoint_eval:02d}.h5"
    )

    out = model.evaluate(x_test, [y_test, x_test])
    logs = pd.DataFrame(np.expand_dims(out, axis=0), columns=model.metrics_names)
    logs.to_csv(
        params["path_results"] / "logs" / f"log_eval-{checkpoint_eval:02d}.csv",
        index=False,
    )

    print(
        f"loss = {out[0]}, "
        f"capsnet_loss = {out[1]}, "
        f"decoder_loss = {out[2]}, "
        f"capsnet_acc = {out[3]}"
    )

    y_prob, recon = model.predict(x_test)
    z_phot = probs_to_redshifts(y_prob, **params)

    met = Metrics(z_phot, z_spec, **params)
    met.full_diagnostic(show=True)
    return out


def predict(model, data, checkpoint_eval, **params):
    """Make predictions.

    Args:
        model: The CapsuleNet model.
        data (array): Images to make predictions for.
        checkpoint (int): Checkpoint number.

    Returns:
        list: loss, capsnet_loss, decoder_loss, capsnet_acc
    """
    model.load_weights(
        params["path_results"] / "weights" / f"weights-{checkpoint_eval:02d}.h5"
    )

    y_prob, recon = model.predict(data)
    y_classes = y_prob.argmax(axis=-1)

    return y_classes, y_prob, recon
