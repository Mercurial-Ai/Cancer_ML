import numpy as np
from math import sqrt
import tensorflow as tf

from models.cancer_ml.image_tools.filter_ids import filter_ids
from models.cancer_ml.image_tools.remove_ids import remove_ids

def import_numpy(path, clinical_ids, random_crop=True, crop_size=(128, 128)):
    img_array = np.load(path)

    img_array = filter_ids(img_array, clinical_ids)

    if random_crop:
        cropped_array = np.empty(shape=(img_array.shape[0], crop_size[0]*crop_size[1]+1), dtype=np.int8)

        i = 0 
        for image in img_array:
            id = image[-1]

            image = remove_ids(image)
            
            image = np.reshape(image, (-1, int(sqrt(len(image)))))

            image = tf.image.random_crop(value=image, size=crop_size)

            image = image.numpy()

            image = image.flatten()

            image = np.append(image, id)

            cropped_array[i] = image

            i = i + 1

        img_array = cropped_array

    return img_array
    