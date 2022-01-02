from sklearn.metrics import confusion_matrix as c_mat 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred):

    if type(y_true) == pd.DataFrame:
        vars = list(y_true.columns)
        y_true = y_true.to_numpy()
    else:
        # if y true is not dataframe, output vars are duke since it is from image only model
        vars = ['Surgery', 'Days to Surgery (from the date of diagnosis)', 'Definitive Surgery Type', 'Clinical Response, Evaluated Through Imaging ', 'Pathologic Response to Neoadjuvant Therapy', 'Days to local recurrence (from the date of diagnosis) ', 'Days to distant recurrence(from the date of diagnosis) ', 'Days to death (from the date of diagnosis) ', 
                            'Days to last local recurrence free assessment (from the date of diagnosis) ', 'Days to last distant recurrence free assemssment(from the date of diagnosis) ', 'Neoadjuvant Chemotherapy', 'Adjuvant Chemotherapy', 'Neoadjuvant Endocrine Therapy Medications ',
                            'Adjuvant Endocrine Therapy Medications ', 'Therapeutic or Prophylactic Oophorectomy as part of Endocrine Therapy ', 'Neoadjuvant Anti-Her2 Neu Therapy', 'Adjuvant Anti-Her2 Neu Therapy ', 'Received Neoadjuvant Therapy or Not', 'Pathologic response to Neoadjuvant therapy: Pathologic stage (T) following neoadjuvant therapy ',
                            'Pathologic response to Neoadjuvant therapy:  Pathologic stage (N) following neoadjuvant therapy', 'Pathologic response to Neoadjuvant therapy:  Pathologic stage (M) following neoadjuvant therapy ', 'Overall Near-complete Response:  Stricter Definition', 'Overall Near-complete Response:  Looser Definition', 'Near-complete Response (Graded Measure)']

    if len(y_true.shape) == 1:
        c_matrix = c_mat(y_true, y_pred.round())
        disp = ConfusionMatrixDisplay(c_matrix)

        plt.close('all')

        disp.plot()

        plt.savefig("confusion_matrix")
        plt.show()

    else:
        y_pred = np.reshape(y_pred, (y_pred.shape[1], y_pred.shape[0]))
        print(y_pred.shape)
        print(y_true.shape)
        for i in range(y_true.shape[1]):
            col1 = y_true[:, i]
            col2 = y_pred[:, i]

            try:
                c_matrix = c_mat(col1, col2.round())
                disp = ConfusionMatrixDisplay(c_matrix)

                # clear plt before plotting to prevent other graphs from appearing with .show()
                plt.close('all')
                
                disp.plot()

                plt.savefig("confusion_matrices/confusion_matrix " + vars[i])
            except ValueError:
                print("var failed in c matrix")
