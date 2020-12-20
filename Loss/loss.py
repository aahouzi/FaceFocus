import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.losses import BinaryCrossentropy, mean_squared_error

cross_entropy = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


def model_vgg19(output_layer=5):
    vgg = VGG19(input_shape=[None, None, 3], include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)


def discriminator_loss(real_output, fake_output, batch_size):
    real_loss = tf.nn.compute_average_loss(cross_entropy(tf.ones_like(real_output), real_output), global_batch_size=batch_size)
    fake_loss = tf.nn.compute_average_loss(cross_entropy(tf.zeros_like(fake_output), fake_output), global_batch_size=batch_size)
    total_loss = real_loss + fake_loss
    return total_loss


def adversarial_loss(fake_output, batch_size):
    return tf.nn.compute_average_loss(cross_entropy(tf.ones_like(fake_output), fake_output), global_batch_size=batch_size)


def content_loss(vgg_model, gen_fake, hr, batch_size):
    # Because we have tanh activation function that maps the values in the range of [-1,1]
    gen_fake, hr = (gen_fake + 1) / 2, (hr + 1) / 2
    gen_fake_features, hr_features = vgg_model(gen_fake), vgg_model(hr)
    return tf.nn.compute_average_loss(mean_squared_error(hr_features, gen_fake_features), global_batch_size=batch_size)







