from src.get_distribution import get_distribution
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def balance_y_classes(x, y, var_to_balance=0, shuffle_data=True):

    if type(x) == pd.DataFrame:
        x = x.to_numpy()

    if type(y) == pd.DataFrame or type(y) == pd.Series:
        y = y.to_numpy()

    if shuffle_data:
        x, y = shuffle(x, y, random_state=31)

    if len(y.shape) == 1:
        var = y
    else:
        var = y[:, var_to_balance]

    var_dist = get_distribution(var)

    overrepresented = max(var_dist.values())

    for class_ in var_dist.keys():
        if var_dist[class_] == overrepresented:
            overrepresented_class = class_

    num_to_remove = x.shape[0]/len(var_dist.values())

    print('overrepresented class:', overrepresented_class)
    print('num to remove:', num_to_remove)

    num_removed = 0
    indices_to_remove = []
    i = 0
    for output in var:
        if num_removed < num_to_remove:
            if output == overrepresented_class:
                indices_to_remove.append(i)
                num_removed = num_removed + 1
        else:
            break

        i = i + 1

    x = np.delete(x, indices_to_remove, axis=0)
    y = np.delete(y, indices_to_remove, axis=0)

    return x, y
        