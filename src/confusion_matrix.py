from sklearn.metrics import confusion_matrix as c_mat 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred):

    if type(y_true) == pd.DataFrame:
        vars = list(y_true.columns)
        print(vars)
        y_true = y_true.to_numpy()

    if len(y_true.shape) == 1:
        c_matrix = c_mat(y_true, y_pred.round())
        disp = ConfusionMatrixDisplay(c_matrix)

        plt.close('all')

        disp.plot()

        plt.savefig("confusion_matrix")
        plt.show()

    else:
        y_pred = np.reshape(y_pred, (y_pred.shape[1], y_pred.shape[0]))
        print(y_true.shape)
        for i in range(y_true.shape[1]):
            col1 = y_true[:, i]
            col2 = y_pred[:, i]

            print(col1.shape)
            print(col2.shape)

            c_matrix = c_mat(col1, col2.round())
            disp = ConfusionMatrixDisplay(c_matrix)

            # clear plt before plotting to prevent other graphs from appearing with .show()
            plt.close('all')
            
            disp.plot()

            plt.savefig("confusion_matrix " + vars[i])
