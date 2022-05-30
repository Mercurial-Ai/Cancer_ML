import numpy as np

def filter_ids(array, img_ids, clinical_ids):

    # list of array indices that need to be deleted
    del_indices = []

    i = 0
    for id in img_ids:
        if id not in clinical_ids:
            del_indices.append(i)

        i = i + 1

    array = np.delete(array, del_indices, axis=-1)

    return array
