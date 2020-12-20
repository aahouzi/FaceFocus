# FaceFocus Project with Colab TPUs | Super Resolution | Computer Vision
Enseirb-Matmeca, Bordeaux INP | [Anas AHOUZI](https://www.linkedin.com/in/anas-ahouzi-6aab0b155/)
***

## :monocle_face: Description
This project aims to implement and train a Super Resolution Generative Adversarial Network (SRGAN), to generate high resolution face images from low resolution ones. The choice of faces is due to the task for which this algorithm can assist, which is enhancing the resolution of face images, by being embedded for example in surveillance cameras
, and help in face recognition tasks.<br />

## :rocket: Repository Structure
The repository contains the following files & directories:
- **Loss directory:** It contains an implementation of various loss functions used for training the SRGAN (Content loss, Adversarial loss, and Perceptual loss).
- **Model directory:** It contains an implementation of the Generator, and the Discriminator architecture of the SRGAN, as described in the original paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf)
- **Train directory:** This directory contains an implementation of the training process used for training the SRGAN with Colab TPUs.
- **Utils directory:** It contains various functions used for visualizing some samples from High/Low resolution images from the dataset.
- **Dataset:** The dataset contains various High/Low resolution face images (about 200 on each), and it's encoded into tfRecord files. It's then placed on a bucket of Google Cloud Storage **due to TPU specific requirements**.



---
## :mailbox_closed: Contact
For any information, feedback or questions, please [contact me][anas-email]










[anas-email]: mailto:ahouzi2000@hotmail.fr