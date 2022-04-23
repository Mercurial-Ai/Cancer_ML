import pandas as pd
import keras
import numpy as np
from itertools import combinations, product
from random import shuffle
from src.grid_search.write_excel import write_excel
from src.confusion_matrix import confusion_matrix
from src.metrics import recall_m, precision_m, f1_m, BalancedSparseCategoricalAccuracy
from tensorflow.keras.metrics import AUC
import torch

class grid_search:

    def read_grid(self):
        grid = pd.read_csv("data/grid.csv")

        grid_cols = list(grid.columns)

        grid_combinations = list(product(grid['batch size'], grid['epochs'], grid['loss'], grid['lr'], grid['optimizer']))
        
        comb_dict_list = []
        for comb in grid_combinations:
            comb = list(comb)

            # default nan_val to false
            nan_val = False

            # check for nans
            for hyp in comb:
                if pd.isnull(hyp):
                    nan_val = True

            if not nan_val:

                comb_dict = dict(zip(grid_cols, comb))

                comb_dict_list.append(comb_dict)

        shuffle(comb_dict_list)

        return comb_dict_list

    def test_model(self, model, X_train, y_train, X_val, y_val, weight_dict=None, num_combs=None):

        if len(X_val) == 1:
            X_val = X_val[0]

        X_val = [torch.from_numpy(X_val[0]).type(torch.float), torch.from_numpy(X_val[1]).type(torch.float)]
        X_val[1] = torch.reshape(X_val[1], (-1, 1, 256, 256))

        combs = self.read_grid()

        print(len(combs), "possible combinations")

        if num_combs == None:
            num_combs = len(combs)

        hyp_acc_list = []
        i = 0
        for comb in combs:
            if i < num_combs:
                auc_m = AUC()
                balanced_acc_m = BalancedSparseCategoricalAccuracy()

                if comb['loss'] == 'mean_absolute_error':
                    self.criterion = torch.nn.L1Loss()
                elif comb['loss'] == 'mean_squared_error':
                    self.criterion = torch.nn.MSELoss()
                elif comb['loss'] == 'binary_crossentropy':
                    self.criterion = torch.nn.CrossEntropyLoss()

                if comb['optimizer'] == 'adam':
                    optimizer = torch.optim.Adam(model.parameters())
                elif comb['optimizer'] == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters())

                model.train_func(X_train, y_train, comb['epochs'], comb['batch size'], optimizer, self.criterion)

                results = [model.loss, float(model.accuracy), float(model.f1_score), float(model.recall), float(model.balanced_acc)]

                confusion_matrix(y_true=y_val, y_pred=model(X_val), save_name="src/grid_search/confusion_matrices/" + str(comb['epochs']) + str(comb['batch size']) + str(comb['loss']) + str(comb['optimizer']) + ".png")

                print("Results:", results)
                loss = results[0]
                percentAcc = results[1]
                f1 = results[2]
                recall = results[3]
                balanced_acc = results[4]

                hyp_acc_pair = (comb, {"Loss": loss, "Accuracy": percentAcc, "F1": f1, "Recall": recall, "Balanced Acc": balanced_acc})

                hyp_acc_list.append(hyp_acc_pair)
                print(hyp_acc_list)
            else:
                break

            i = i + 1

        writer = write_excel('results.xls', hyp_acc_list)
        writer.run()
