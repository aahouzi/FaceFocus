
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import glob


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
    return lambda x: tf.nn.depth_to_space(x, scale)


def _bytes_feature(value):
    """Convert string / byte in bytes_list."""
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
    image_shape = tf.image.decode_png(image_string).shape

    feature = {'height': _int64_feature(image_shape[0]),
               'width': _int64_feature(image_shape[1]),
               'depth': _int64_feature(image_shape[2]),
               'image': _bytes_feature(image_string)
               }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def to_tfrecord(record_file, path_to_folders):
    with tf.io.TFRecordWriter(record_file) as writer:
        filenames = glob.glob(f"{path_to_folders}/*.png")
        for file in filenames:
            image_string = open(file, 'rb').read()
            # Convert image in `tf.Example`- message
            tf_example = image_example(image_string)
            # Then write in '.tfrecords' file
            writer.write(tf_example.SerializeToString())


def get_dataset(tfrecord_file, hr_shape, lr_shape, features_description, batch_size, buffer_size=2):
    image_dataset = tf.data.TFRecordDataset(tfrecord_file)

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        feature = tf.io.parse_single_example(example_proto, features_description)
        image = tf.image.decode_png(feature['image'], channels=3)
        # Get the required sizes for SRGAN training
        hr_image = tf.cast(tf.image.resize(image, size=[*hr_shape], method='bicubic'), dtype=tf.float32)
        lr_image = tf.cast(tf.image.resize(image, size=[*lr_shape], method='bicubic'), dtype=tf.float32)

        return hr_image, lr_image

    dataset = image_dataset.map(_parse_image_function, num_parallel_calls=multiprocessing.cpu_count()) \
                           .shuffle(128).repeat().batch(batch_size).prefetch(buffer_size)

    return dataset


def show_samples(tfrecord_file, hr_shape, lr_shape, features_description, batch_size):
    dataset = get_dataset(tfrecord_file, hr_shape, lr_shape, features_description, batch_size)
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


def resolve(model, batch_lr):
    batch_lr = tf.cast(batch_lr, tf.float32)
    batch_sr = model(batch_lr, training=False)
    batch_sr = tf.clip_by_value(batch_sr, 0, 255)
    batch_sr = tf.round(batch_sr)
    batch_sr = tf.cast(batch_sr, tf.uint8)
    return batch_sr


def generate_and_save_images(model, batch_lr, epoch, batch_size=4):
    batch_sr = resolve(model, batch_lr)
    figure, axes = plt.subplots(nrows=1, ncols=4, figsize=(140, 140))

    for i in range(batch_size):
        axes[i].imshow(batch_sr[i], interpolation='nearest')

    plt.savefig('image_at_epoch_{}.png'.format(epoch))

    return batch_sr











