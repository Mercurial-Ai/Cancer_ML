import pandas as pd
from itertools import product
import math

class grid_search:
    def __init__(self, grid_path):
        self.grid_path = grid_path
    
    def read_grid(self):
        grid = pd.read_csv(self.grid_path)

        grid_cols = list(grid.columns)

        grid_combs = list(product(grid['batch size'], grid['epochs'], grid['loss'], grid['optimizer']))

        comb_dict_list = []
        for comb in grid_combs:
            comb = list(comb)
            
            # check for nans
            for hyp in comb:
                if str(type(hyp)) != "<class 'str'>":
                    if math.isnan(hyp):
                        nan_val = True
                    else:
                        nan_val = False
                else:
                    nan_val = False

            if not nan_val:

                comb_dict = dict(zip(grid_cols, comb))

                comb_dict_list.append(comb_dict)

        return comb_dict_list
