############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: Utils/utils.py                                  #
#                           Creation Date: December 17, 2019                               #
#                         Source Language: Python                                          #
#                  Repository: https://github.com/aahouzi/FaceFocus.git                    #
#                              --- Code Description ---                                    #
#          Implementation of various Loss functions used for training the SRGAN            #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.losses import BinaryCrossentropy, mean_squared_error


################################################################################
#                                  Main Code                                   #
################################################################################

# Define the binary crossentropy loss
cross_entropy = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


def model_vgg19(output_layer=5):
    """Returns a pre-trained VGG19 model"""
    vgg = VGG19(input_shape=[None, None, 3], include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)


def discriminator_loss(real_output, fake_output, batch_size):
    """
    Computes the discriminator loss after training with HR & fake images.
    :param real_output: Discriminator output of the real dataset (HR images).
    :param fake_output: Discriminator output of the fake dataset (SR images).
    :param batch_size: Batch size.
    :return: Discriminator loss.
    """
    real_loss = tf.nn.compute_average_loss(cross_entropy(tf.ones_like(real_output), real_output),
                                           global_batch_size=batch_size)
    fake_loss = tf.nn.compute_average_loss(cross_entropy(tf.zeros_like(fake_output), fake_output),
                                           global_batch_size=batch_size)
    total_loss = real_loss + fake_loss
    return total_loss


def adversarial_loss(fake_output, batch_size):
    """
    Computes the adversarial loss.
    :param fake_output: Discriminator output of the fake dataset (SR images).
    :param batch_size: Batch size.
    :return: Adversarial loss.
    """
    return tf.nn.compute_average_loss(cross_entropy(tf.ones_like(fake_output), fake_output),
                                      global_batch_size=batch_size)


def content_loss(vgg_model, sr, hr, batch_size):
    """
    Computes the VGG-19 based content loss.
    :param vgg_model: A pre-trained VGG-19 model.
    :param sr: Super resolution images.
    :param hr: High resolution images.
    :param batch_size: Batch size.
    :return: VGG-19 based content loss.
    """
    sr_features, hr_features = vgg_model(sr), vgg_model(hr)
    # Scale VGG19 feature maps, to obtain a loss comparable to MSE loss
    sr_features, hr_features = sr_features / 12.75, hr_features / 12.75
    return tf.nn.compute_average_loss(mean_squared_error(hr_features, sr_features), global_batch_size=batch_size)








