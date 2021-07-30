import pandas as pd
from itertools import product
from random import shuffle

class grid_search:
    def __init__(self, grid_path):
        self.grid_path = grid_path
    
    def read_grid(self):
        grid = pd.read_csv(self.grid_path)

        grid_cols = list(grid.columns)

        grid_combs = list(product(grid['batch size'], grid['epochs'], grid['loss'], grid['lr'], grid['optimizer']))

        comb_dict_list = []
        for comb in grid_combs:
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
