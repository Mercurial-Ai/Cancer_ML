import pandas as pd

# return dictionary of distribution of array's elements
def get_distribution(iterable):

    if type(iterable) != pd.Series:
        iterable = pd.Series(iterable)

    counts_dict = iterable.value_counts().to_dict()

    percent_list = []
    for count in list(counts_dict.values()):
        percent = (count/len(iterable))*100
        percent = round(percent, 0)
        percent_list.append(percent)

    count_keys = list(counts_dict.keys())

    percent_dict = dict(zip(count_keys, percent_list))

    return percent_dict
