import numpy as np
from tensorflow.keras import layers, models

from ..base.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


def CapsNet(
    input_shape,
    num_class,
    dim_capsule,
    routings,
    learning_rate,
    lam_recon,
    **kwargs,
):
    """Capsule Network for MNIST.

    Args:
        input_shape (tuple): Data shape [width, height, channels].
        num_class (int): Number of classes.
        dim_capsule (int): Dimesion of capsule.
        routings (int): Number of routing iterations.
        learning_rate (float): Learning rate.
        decay_rate (float): Decay of learning rate.
        lam_recon (float): Reconstruction regularization coefficient.

    Returns:
        Keras Model, Keras Model: ``train_model`` used for training,
            ``eval_model`` for evaluation.
    """

    # The Encoder Networks

    # input layer
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Another conventional Conv2D layer
    conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu', name='conv2')(conv1)

    # Layer 2.5: Avg Pooling layer
    avg25 = layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(conv2)

    # Layer 3: Another conventional Conv2D layer
    conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid', activation='relu', name='conv3')(avg25)

    # Layer 4: Primary Caps Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv3, dim_capsule=dim_capsule, n_channels=8, kernel_size=(3, 3), strides=2, padding='valid')

    # Layer 5: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=num_class, dim_capsule=dim_capsule, routings=3, name='digitcaps')(primarycaps)

    # Layer 5.5: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(num_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=dim_capsule * num_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model
