import tensorflow as tf
import numpy as np

def random_crop(image_array, crop_size):

    new_array = np.empty(shape=(len(image_array), crop_size[0], crop_size[1], 1), dtype=np.int8)

    i = 0
    for image in image_array:
        
        image = tf.image.random_crop(value=image, size=crop_size)
        image = image.numpy()
        new_array[i] = image

        i = i + 1
    
    return new_array
