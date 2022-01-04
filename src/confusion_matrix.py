from sklearn.metrics import confusion_matrix as c_mat 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred):

    c_matrix = c_mat(y_true, y_pred.round())
    disp = ConfusionMatrixDisplay(c_matrix)

    plt.close('all')
    
    disp.plot()

    plt.savefig("confusion_matrix" + str(y_true.shape))
