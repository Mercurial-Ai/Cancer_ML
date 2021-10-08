import numpy as np

def get_percentage_accuracy(y_pred, y_true):

    y_pred = np.array(y_pred)

    y_pred = np.around(y_pred, decimals=0)

    num_correct = len([i for i, j in zip(y_pred, y_true) if i == j])

    percent_correct = num_correct/len(y_pred)
    percent_correct = percent_correct * 100

    percent_correct = round(percent_correct, 2)

    return percent_correct
    