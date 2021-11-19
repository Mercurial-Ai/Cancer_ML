import pandas as pd
import keras
import numpy as np
from itertools import combinations, product
from random import shuffle

from write_excel import write_excel

class grid_search:

    def read_grid(self):
        grid = pd.read_csv("grid.csv")

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

    def test_model(self, model, X_train, y_train, X_test, y_test, weight_dict):

        X_train = np.load(X_train)
        y_train = np.load(y_train)
        X_test = np.load(X_test)
        y_test = np.load(y_test)

        combs = self.read_grid()

        for comb in combs:
            model_copy = model
            model_copy.compile(loss=comb['loss'], optimizer=comb['optimizer'], metrics=['accuracy', ''])

            fit = model_copy.fit(X_train, y_train, epochs=comb['epochs'], batch_size=comb['batch size'], class_weight=weight_dict)

            results = model_copy.evaluate(X_test, y_test, batch_size=comb['batch size'])

            writer = write_excel('results', results)
            writer.run()
