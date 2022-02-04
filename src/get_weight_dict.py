from src.get_distribution import get_distribution
import pandas as pd

def get_weight_dict(array):

    if type(array) == pd.DataFrame:
        array = array.to_numpy()

    weight_dict={}
    if len(array.shape) > 1:
        for i in range(array.shape[-1]):
            var = array[:, i]

            array_dict = get_distribution(var)

            keys = list(array_dict.keys())

            # reverse values of dict
            values = list(array_dict.values())
            values.reverse()

            weight_dict["dense_" + str(6+i)] = values

    else:
        array_dict = get_distribution(array)

        keys = list(array_dict.keys())

        # reverse values of dict
        values = list(array_dict.values())
        values.reverse()

        weight_dict = dict(zip(keys, values))

    return weight_dict
    