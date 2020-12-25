############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: Utils/utils.py                                  #
#                           Creation Date: December 17, 2019                               #
#                         Source Language: Python                                          #
#                  Repository: https://github.com/aahouzi/FaceFocus.git                    #
#                              --- Code Description ---                                    #
#                             Various utility functions                                    #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import glob


################################################################################
#                                  Main Code                                   #
################################################################################
def normalize(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_tanh(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_tanh(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


def pixel_shuffle(scale):
    """It's used in the upscale blocks to enhance the image resolution"""
    return lambda x: tf.nn.depth_to_space(x, scale)


def _bytes_feature(value):
    """Convert string/byte in bytes_list."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Convert float / double in float_list."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Convert bool / enum / int / uint in int64_list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string):
    """Generates a tf.train.Example sample to create tfRecord files"""
    image_shape = tf.image.decode_png(image_string).shape

    feature = {'height': _int64_feature(image_shape[0]),
               'width': _int64_feature(image_shape[1]),
               'depth': _int64_feature(image_shape[2]),
               'image': _bytes_feature(image_string)
               }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def to_tfrecord(record_file, path_to_folders):
    """
    Generates tfRecord files.
    :param record_file: Path to location where to put the generated tfRecords.
    :param path_to_folders: Path to Train/Validation dataset.
    :return:
    """
    with tf.io.TFRecordWriter(record_file) as writer:
        filenames = glob.glob(f"{path_to_folders}/*.png")
        for file in filenames:
            image_string = open(file, 'rb').read()
            # Convert image in `tf.Example`- message
            tf_example = image_example(image_string)
            # Then write in '.tfrecords' file
            writer.write(tf_example.SerializeToString())


def get_dataset(tfrecord_file, hr_shape, lr_shape, batch_size, buffer_size=2):
    """
    Returns a tf.data.Dataset object from a tfRecord file.
    :param tfrecord_file: Path to tfRecord file.
    :param hr_shape: Shape of high resolution images.
    :param lr_shape: Shape of low resolution images.
    :param batch_size: Size of a batch of images in the dataset.
    :param buffer_size: The maximum number of elements that will be buffered when prefetching.
    :return: A tf.data.Dataset object.
    """

    # Read the tfRecord file
    image_dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Description of features in tf.train.Example
    features_description = {'height': tf.io.FixedLenFeature([], tf.int64),
                            'width': tf.io.FixedLenFeature([], tf.int64),
                            'depth': tf.io.FixedLenFeature([], tf.int64),
                            'image': tf.io.FixedLenFeature([], tf.string)
                            }

    def parse_image_function(example_proto):
        """This function parses a single example proto from the tfRecord file."""
        # Parse the input tf.Example proto using the dictionary above.
        feature = tf.io.parse_single_example(example_proto, features_description)
        image = tf.image.decode_png(feature['image'], channels=3)
        # Get the required sizes for SRGAN training
        hr_image = tf.cast(tf.image.resize(image, size=hr_shape, method='bicubic'), dtype=tf.float32)
        lr_image = tf.cast(tf.image.resize(image, size=lr_shape, method='bicubic'), dtype=tf.float32)

        return hr_image, lr_image

    # Create the tf.data.Dataset object
    dataset = image_dataset.map(parse_image_function, num_parallel_calls=multiprocessing.cpu_count()) \
                           .shuffle(128).repeat().batch(batch_size).prefetch(buffer_size)

    return dataset


def show_samples(tfrecord_file, hr_shape, lr_shape, batch_size):
    """
    Visualize a batch of HR/LR images from the dataset.
    :param tfrecord_file: Path to tfRecord file.
    :param hr_shape: Shape of high resolution images.
    :param lr_shape: Shape of low resolution images.
    :param batch_size: Size of a batch of images in the dataset.
    :return:
    """
    dataset = get_dataset(tfrecord_file, hr_shape, lr_shape, batch_size)
    figure, axes = plt.subplots(nrows=2, ncols=4, figsize=(140, 140))

    for hr, lr in dataset.unbatch().batch(batch_size).take(1):
        # We get a batch of 4 images of shape img_shape (4, img_shape)
        hr, lr = tf.cast(hr, tf.uint8), tf.cast(lr, tf.uint8)
        for i in range(2):
            for j in range(batch_size):
                if i == 0:
                    axes[i, j].imshow(hr[j], interpolation='nearest')
                else:
                    axes[i, j].imshow(lr[j], interpolation='nearest')

    return hr, lr


def generate_and_prepare(model, batch_lr):
    """This function generates a SR batch, and prepares it for visualization"""
    batch_lr = tf.cast(batch_lr, tf.float32)
    batch_sr = model(batch_lr, training=False)
    batch_sr = tf.clip_by_value(batch_sr, 0, 255)
    batch_sr = tf.round(batch_sr)
    batch_sr = tf.cast(batch_sr, tf.uint8)
    return batch_sr


def plot_and_save(model, batch_lr, drive_path, epoch, batch_size=4):
    """
    Displays the generated SR batch, and save it to the drive.
    :param model: The generator model.
    :param batch_lr: A batch of LR images.
    :param drive_path: Path to a drive directory for saving images.
    :param epoch: Number of the actual epoch.
    :param batch_size: Batch size.
    :return: Returns a SR batch, used to compute PSNR/SSIM metrics.
    """
    batch_sr = generate_and_prepare(model, batch_lr)
    figure, axes = plt.subplots(nrows=1, ncols=4, figsize=(140, 140))

    for i in range(batch_size):
        axes[i].imshow(batch_sr[i], interpolation='nearest')

    plt.savefig(drive_path+'/image_every_100epoch/image_at_epoch_{}.png'.format(epoch))

    return batch_sr











