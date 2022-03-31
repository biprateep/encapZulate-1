import numpy as np

from keras import backend as K
from keras import layers, models, optimizers
from keras.layers import Layer
from keras.layers import (
    Input,
    Conv2D,
    Activation,
    Dense,
    Dropout,
    Lambda,
    Reshape,
    Concatenate,
)
from keras.layers import (
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Conv1D,
    Deconvolution2D,
    Conv2DTranspose,
    Softmax,
)
from ..base.deepCapsLayers import (
    Conv2DCaps,
    ConvCapsuleLayer3D,
    CapsuleLayer,
    CapsToScalars,
    Mask_CID,
    ConvertToCaps,
    FlattenCaps,
)


def CapsNet(input_shape, num_class, routings, dim_capsule, **kwargs):
    # assemble encoder
    x = Input(shape=input_shape)
    l = x

    l = Conv2D(
        128,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(
        l
    )  # common conv layer
    l = BatchNormalization()(l)
    l = ConvertToCaps()(l)

    l = Conv2DCaps(
        32, 4, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l_skip = Conv2DCaps(
        32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = Conv2DCaps(
        32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = Conv2DCaps(
        32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l_skip = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l_skip = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = layers.Add()([l, l_skip])
    l1 = l

    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l_skip = ConvCapsuleLayer3D(
        kernel_size=3,
        num_capsule=32,
        num_atoms=8,
        strides=1,
        padding="same",
        routings=3,
    )(l)
    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = Conv2DCaps(
        32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1]
    )(l)
    l = layers.Add()([l, l_skip])
    l2 = l

    la = FlattenCaps()(l2)
    lb = FlattenCaps()(l1)
    l = layers.Concatenate(axis=-2)([la, lb])

    #     l = Dropout(0.4)(l)
    digits_caps = CapsuleLayer(
        num_capsule=num_class,
        dim_capsule=dim_capsule,
        routings=routings,
        channels=0,
        name="digit_caps",
    )(l)

    l = CapsToScalars(name="capsnet")(digits_caps)
    # l = Softmax()(l)

    m_capsnet = models.Model(inputs=x, outputs=l, name="capsnet_model")

    y = Input(shape=(num_class,))

    masked_by_y = Mask_CID()([digits_caps, y])
    masked = Mask_CID()(digits_caps)

    # Decoder Network
    decoder_input = Input(shape=(dim_capsule,))
    d = Dense(np.prod(input_shape), activation="relu", kernel_initializer="he_normal",)(
        decoder_input
    )
    d = Reshape(input_shape)(d)

    d = Conv2DTranspose(
        64, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal",
    )(d)
    d = Conv2DTranspose(
        32, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal",
    )(d)
    d = Conv2DTranspose(
        16, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal",
    )(d)

    d = Conv2DTranspose(
        8, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal"
    )(d)

    d = Conv2DTranspose(
        input_shape[-1], (3, 3), padding="same", activation="sigmoid", kernel_initializer="he_normal",
    )(d)
    decoder_output = Reshape(target_shape=input_shape, name="out_recon")(d)

    decoder = models.Model(decoder_input, decoder_output, name="decoder_model")
    train_model = models.Model([x, y], [m_capsnet.output, decoder(masked_by_y)])

    eval_model = models.Model(x, [m_capsnet.output, decoder(masked)])
    # manipulate_model = models.Model(x, [digits_caps, m_capsnet.output, decoder(masked)])
    manipulate_model = models.Model(x, [masked, m_capsnet.output, decoder(masked)])
    train_model.summary()

    return train_model, eval_model, manipulate_model, decoder
