############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: Model/srgan.py                                  #
#                           Creation Date: December 17, 2019                               #
#                         Source Language: Python                                          #
#                  Repository: https://github.com/aahouzi/FaceFocus.git                    #
#                              --- Code Description ---                                    #
#                       Implementation of the SRGAN architecture                           #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.keras.models import Model
from Utils.utils import pixel_shuffle, normalize


################################################################################
#                                 Main Code                                    #
################################################################################
def upsample(x_in, num_filters):
    """
    Upsampling block, used to upsample the image in the generator.
    :param x_in: A tf.tensor.
    :param num_filters: Number of filters in the 2D convolution layer.
    :return: A tf.tensor.
    """
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def residual_block(x_in, num_filters, momentum=0.8):
    """
    A residual block used in the generator.
    :param x_in: A tf.tensor.
    :param num_filters: Number of filters in the 2D convolution layer.
    :param momentum: A hyper-parameter for BatchNorm layer.
    :return: A tf.tensor.
    """
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    """
    A block of (2D convolution layer + BatchNorm) used in the discriminator.
    :param x_in: A tf.tensor.
    :param num_filters: Number of filters in the 2D convolution layer.
    :param strides: Number of columns by which the sliding window moves in 2D CONV layer.
    :param batchnorm: Whether to apply or not BatchNormalization (1 or 0).
    :param momentum: A hyper-parameter used for BatchNorm layer.
    :return: A tf.tensor.
    """
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def generator(lr_size, num_filters=64, num_res_blocks=16):
    """
    An implementation of the basic architecture of a generator in the SRGAN.
    :param lr_size: Size of low resolution images.
    :param num_filters: A multiple of the number of filters in 2D convolution layers.
    :param num_res_blocks: Number of residual blocks.
    :return: A Keras functional API model.
    """
    x_in = Input(shape=(lr_size, lr_size, 3))

    # Scale the values to range [0, 1]
    x = Lambda(normalize)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = residual_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4)
    x = upsample(x, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)

    return Model(x_in, x)


def discriminator(hr_size, num_filters=64):
    """
    An implementation of the basic architecture of a discriminator in the SRGAN.
    :param hr_size: Size of high resolution images.
    :param num_filters: A multiple of the number of filters in 2D convolution layers.
    :return: A Keras functional API model.
    """
    x_in = Input(shape=(hr_size, hr_size, 3))

    x = discriminator_block(x_in, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


