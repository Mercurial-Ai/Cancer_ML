from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(c_matrix)

    disp.plot()

    plt.savefig("confusion_matrix")
    plt.show()
