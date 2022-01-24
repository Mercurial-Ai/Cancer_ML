import numpy as np
import pandas as pd

from src.get_distribution import get_distribution

def equalize_clinical_classes(data, target, imagery=None):

    if type(data) == pd.DataFrame:
        target = data[target]
    else:
        target = data[:, target]

    target_distribution = get_distribution(target)

    classes = list(target_distribution.keys())
    dists = list(target_distribution.values())

    largest_dist = max(dists)
    smallest_dist = min(dists)

    dist_index = dists.index(largest_dist)

    largest_class = classes[dist_index]

    classes_to_delete = (len(target)*largest_dist*0.01) - (len(target)*smallest_dist*0.01)

    i = 0
    classes_added = 0
    indices_to_delete = []
    for t in target:
        if classes_added < classes_to_delete:
            if t == largest_class:
                classes_added = classes_added + 1
                indices_to_delete.append(i)

        i = i + 1

    indices_not_delete = []
    for i in range(len(target)):
        if i not in indices_to_delete:
            indices_not_delete.append(i)

    if type(data) == pd.DataFrame:
        clinical_data = data.iloc[indices_not_delete]
    else:
        clinical_data = data[indices_not_delete]
        
    if type(imagery) != type(None):
        imagery = imagery[indices_not_delete]

    return clinical_data, imagery
