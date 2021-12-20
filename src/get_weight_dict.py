from src.get_distribution import get_distribution

def get_weight_dict(array):
    array_dict = get_distribution(array)

    keys = list(array_dict.keys())

    # reverse values of dict
    values = list(array_dict.values())
    values.reverse()

    weight_dict = dict(zip(keys, values))

    return weight_dict
    