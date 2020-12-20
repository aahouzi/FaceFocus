############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: Train/TPU_training.py                           #
#                           Creation Date: December 17, 2019                               #
#                         Source Language: Python                                          #
#                  Repository: https://github.com/aahouzi/FaceFocus-Project                #
#                              --- Code Description ---                                    #
#                 Implementation of TPU training process for the SRGAN                     #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Model.srgan import generator, discriminator
from Loss.loss import model_vgg19, content_loss, adversarial_loss, discriminator_loss
from Utils.utils import get_dataset, generate_and_save_images
from collections import defaultdict
import numpy as np
import argparse
import time
import os


################################################################################
#                                Main arguments                                #
################################################################################

parser = argparse.ArgumentParser(description='Train the model using Colab TPUs.')

parser.add_argument('--n_epochs', required=True, help='Number of epochs')
parser.add_argument('--hr_shape', required=True, help='High resolution shape')
parser.add_argument('--lr_shape', required=True, help='Low resolution shape')
parser.add_argument('--train_hr_path', required=True, help='Path to training HR tfRecords in Google Cloud Storage')
parser.add_argument('--val_hr_path', required=True, help='Path to validation HR tfRecords in Google Cloud Storage')
parser.add_argument('--batch_val_hr', required=True, help='A batch of 4 HR validation images')
parser.add_argument('--batch_val_lr', required=True, help='A batch of 4 LR validation images')
parser.add_argument('--features_description', required=True, help='Description of a tf.train.Example in tfRecords')

args = parser.parse_args()

################################################################################
#                                   Main code                                  #
################################################################################


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('\n\n'+'---' * 10 + 'TPU WORKS' + '---' * 10)

elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])

elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)

else:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')

print("\n Number of accelerators: {}\n".format(strategy.num_replicas_in_sync))


with strategy.scope():
    # Get the Generator/Discriminator
    discriminator_model = discriminator(hr_size=256)
    generator_model = generator(lr_size=64)

    # Load the pre weights
    # generator_model.load_weights(r'/sample_data/pre_generator.h5')

    # Define the loss function for the discriminator, and the optimizer
    VGG = model_vgg19()
    optimizer = Adam(learning_rate=0.001)

    # Instantiate metrics
    adversarial_loss_sum = tf.keras.metrics.Sum()
    discriminator_loss_sum = tf.keras.metrics.Sum()
    content_loss_sum = tf.keras.metrics.Sum()
    perceptual_loss_sum = tf.keras.metrics.Sum()


    @tf.function
    def train_step(dataset, batch_size):
        """
        This function performs one training step on a batch of HR/LR images.
        :param dataset:
        :param batch_size:
        :return:
        """
        # Get HR/LR images
        hr_img, lr_img = dataset

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Start the training
            generated_images = generator_model(lr_img, training=True)
            fake_output = discriminator_model(generated_images, training=True)
            real_output = discriminator_model(hr_img, training=True)

            # Compute generator loss
            cont_loss = content_loss(VGG, generated_images, hr_img, batch_size)
            adv_loss = adversarial_loss(fake_output, batch_size)
            perceptual_loss = cont_loss + 1e-3 * adv_loss

            # Compute discriminator loss
            disc_loss = discriminator_loss(real_output, fake_output, batch_size)

        # Compute the gradient of the discriminator
        grads_disc = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
        optimizer.apply_gradients(zip(grads_disc, discriminator_model.trainable_variables))

        # Compute the gradient of the generator
        grads_gen = gen_tape.gradient(perceptual_loss, generator_model.trainable_variables)
        optimizer.apply_gradients(zip(grads_gen, generator_model.trainable_variables))

        # update metrics
        discriminator_loss_sum.update_state(disc_loss)
        adversarial_loss_sum.update_state(adv_loss)
        content_loss_sum.update_state(cont_loss)
        perceptual_loss_sum.update_state(perceptual_loss)


    # Distribute the dataset according to the strategy.
    train_dataset = get_dataset(args.train_hr_path, args.hr_shape, args.lr_shape, args.features_description,
                                batch_size=4 * strategy.num_replicas_in_sync)

    # Returns a tf.distribute.DistributedDataset from tf.data.Dataset, which represents a dataset
    # distributed among devices and machines.
    train_dist_img = strategy.experimental_distribute_dataset(train_dataset)

    # Define the batch size based on number of cores.
    batch_size = tf.constant(4 * strategy.num_replicas_in_sync, dtype=tf.float32)

    # Define Number of steps per epoch.
    steps_per_epoch = 800 // (4 * strategy.num_replicas_in_sync)

    print("\n[INFO]: Steps per epoch: {}".format(steps_per_epoch))
    Loss = defaultdict(list)
    epoch_start_time = time.time()
    epoch = 0
    # A step is one gradient update over a batch of data, An epoch
    # is usually many steps when we go through the whole training set.
    for step, images in enumerate(train_dist_img):

        # Launch the training
        strategy.run(train_step, args=(images, batch_size))
        print('=', end='', flush=True)

        # Displaying the results after each epoch
        if ((step + 1) // steps_per_epoch) > epoch:
            print('>', end='', flush=True)

            # compute metrics
            Loss['adv_loss'].append(adversarial_loss_sum.result().numpy() / steps_per_epoch)
            Loss['disc_loss'].append(discriminator_loss_sum.result().numpy() / steps_per_epoch)
            Loss['perceptual_loss'].append(perceptual_loss_sum.result().numpy() / steps_per_epoch)
            Loss['content_loss'].append(content_loss_sum.result().numpy() / steps_per_epoch)

            if epoch % 100 == 0:
                # Test the model over a batch of 4 images (batch_lr), and save the results
                batch_sr = generate_and_save_images(generator_model, args.batch_val_lr, epoch)

                # Compute the PSNR/SSIM metrics
                psnr_metric = round(np.sum(tf.image.psnr(batch_sr, args.batch_val_hr, max_val=255.0)) / 4, 2)
                ssim_metric = round(np.sum(tf.image.ssim(batch_sr, args.batch_val_hr, max_val=255.0)) / 4, 2)

                print('\n PSNR: {} | SSIM: {} \n'.format(psnr_metric, ssim_metric))

                # Save the models every 100 epoch
                generator_model.save('ModelTPU-generator-{}.h5'.format(epoch))
                discriminator_model.save('ModelTPU-discriminator-{}.h5'.format(epoch))

            # report metrics
            duration = round(time.time() - epoch_start_time, 2)
            print('\nEpoch: {}/{}'.format(epoch, args.n_epochs),
                  'Duration: {}s'.format(duration),
                  'disc_loss: {}'.format(round(Loss['disc_loss'][-1], 2)),
                  'adv_loss: {}'.format(round(Loss['adv_loss'][-1], 2)),
                  'content_loss: {}'.format(round(Loss['content_loss'][-1], 2)),
                  'perceptual_loss: {}'.format(round(Loss['perceptual_loss'][-1], 2)),
                  flush=True)

            # set up next epoch
            epoch = (step + 1) // args.steps_per_epoch
            epoch_start_time = time.time()
            adversarial_loss_sum.reset_states()
            discriminator_loss_sum.reset_states()
            content_loss_sum.reset_states()
            perceptual_loss_sum.reset_states()

            if epoch >= args.n_epochs:
                break

