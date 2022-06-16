import numpy as np
import tensorflow as tf

def four_dim_normalization(img_array):

    i = 0 
    for p in img_array:
        s = tf.image.per_image_standardization(p)
        img_array[i] = s
        i = i + 1
        
    return img_array
