# 0x04 Data Augmentation

> Data augmentation is a process of artificially increasing the amount of data by generating new data points from existing data. This includes adding minor alterations to data or using machine learning models to generate new data points in the latent space of original data to amplify the dataset

At the end of this project I was able to answer these conceptual questions:

* What is data augmentation?
* When should you perform data augmentation?
* What are the benefits of using data augmentation?
* What are the various ways to perform data augmentation?
* How can you use ML to automate data augmentation?

## Tasks

0. Write a function `def flip_image(image):` that flips an image horizontally:

    * `image` is a 3D `tf.Tensor` containing the image to flip
    * Returns the flipped image

1. Write a function `def crop_image(image, size):` that performs a random crop of an image:

    * `image` is a 3D `tf.Tensor` containing the image to crop
    * `size` is a tuple containing the size of the crop
    * Returns the cropped image

2. Write a function `def rotate_image(image):` that rotates an image by 90 degrees counter-clockwise:

    * `image` is a 3D `tf.Tensor` containing the image to rotate
    * Returns the rotated image

3. Write a function `def shear_image(image, intensity):` that randomly shears an image:

    * `image` is a 3D `tf.Tensor` containing the image to shear
    * `intensity` is the intensity with which the image should be sheared
    * Returns the sheared image

4. Write a function `def change_brightness(image, max_delta):` that randomly changes the brightness of an image:

    * `image` is a 3D `tf.Tensor` containing the image to change
    * `max_delta` is the maximum amount the image should be brightened (or darkened)
    * Returns the altered image

5. Write a function `def change_hue(image, delta):` that changes the hue of an image:

    * `image` is a 3D `tf.Tensor` containing the image to change
    * `delta` is the amount the hue should change
    * Returns the altered image

