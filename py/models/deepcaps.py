import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Dropout, Lambda, Reshape, Concatenate
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Flatten, Conv1D, Deconvolution2D, Conv2DTranspose, Softmax
from ..base.deepCapsLayers import Conv2DCaps, ConvCapsuleLayer3D, CapsuleLayer, CapsToScalars, Mask_CID, ConvertToCaps, FlattenCaps


def CapsNet(input_shape, num_class, routings,dim_capsule,**kwargs):
    # assemble encoder
    x = Input(shape=input_shape)
    l = x

    l = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding="same",kernel_initializer='he_normal')(l)  # common conv layer
    l = BatchNormalization()(l)
    l = ConvertToCaps()(l)

    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])
    l1 = l

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = ConvCapsuleLayer3D(kernel_size=3, num_capsule=32, num_atoms=8, strides=1, padding='same', routings=3)(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])
    l2 = l


    la = FlattenCaps()(l2)
    lb = FlattenCaps()(l1)
    l = layers.Concatenate(axis=-2)([la, lb])

#     l = Dropout(0.4)(l)
    digits_caps = CapsuleLayer(num_capsule=num_class, dim_capsule=dim_capsule, routings=routings, channels=0, name='digit_caps')(l)

    l = CapsToScalars(name='capsnet')(digits_caps)
    #l = Softmax()(l)

    m_capsnet = models.Model(inputs=x, outputs=l, name='capsnet_model')

    y = Input(shape=(num_class,))

    masked_by_y = Mask_CID()([digits_caps, y])  
    masked = Mask_CID()(digits_caps)

    # Decoder Network
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(input_dim=dim_capsule, activation="relu",kernel_initializer='he_normal', output_dim=np.prod(input_shape)))
    decoder.add(Reshape(input_shape))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Conv2DTranspose(64, (3,3), padding='same',activation="relu",kernel_initializer='he_normal'))
    decoder.add(Conv2DTranspose(32, (3,3), padding='same', activation="relu",kernel_initializer='he_normal'))
    decoder.add(Conv2DTranspose(16, (3,3), padding='same',activation="relu",kernel_initializer='he_normal'))
    decoder.add(Conv2DTranspose(8, (3,3), padding='same', activation="relu",kernel_initializer='he_normal'))
    decoder.add(Conv2DTranspose(5, (3,3), padding='same',activation="linear",kernel_initializer='he_normal'))
    decoder.add(Reshape(target_shape=(64, 64, 5), name='out_recon'))


    train_model = models.Model([x, y], [m_capsnet.output, decoder(masked_by_y)])
    eval_model = models.Model(x, [m_capsnet.output, decoder(masked)])
    train_model.summary()

    return train_model, eval_model
