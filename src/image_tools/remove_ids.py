import numpy as np

def remove_ids(img_array):

    try:
        new_shape = (img_array.shape[0], img_array.shape[1]-1)
        new_array = np.empty(shape = new_shape, dtype=np.int8)

        i = 0
        for img in img_array:
            img = np.delete(img, -1)
            new_array[i] = img
            i = i + 1
    except IndexError:
        new_shape = img_array.shape

        img_array = np.delete(img_array, -1)

        new_array = img_array

    return new_array
    