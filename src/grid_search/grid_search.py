import pandas as pd
import keras
import numpy as np
from itertools import combinations, product
from random import shuffle

from src.grid_search.write_excel import write_excel

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

        combs = self.read_grid()

        print(len(combs), "possible combinations")

        if num_combs == None:
            num_combs = len(combs)

        hyp_acc_list = []
        i = 0
        for comb in combs:
            if i < num_combs:
                model_copy = model
                model_copy.compile(loss=comb['loss'], optimizer=comb['optimizer'], metrics=['accuracy'])

                fit = model_copy.fit(X_train, y_train, epochs=comb['epochs'], batch_size=comb['batch size'], validation_data=(X_val, y_val), class_weight=weight_dict)

                results = model_copy.evaluate(X_val, y_val, batch_size=comb['batch size'])

                percentAcc = results[-1]

                hyp_acc_pair = (comb, (percentAcc))

                hyp_acc_list.append(hyp_acc_pair)
                print(hyp_acc_list)
            else:
                break

            i = i + 1

        writer = write_excel('results.xls', hyp_acc_list)
        writer.run()
