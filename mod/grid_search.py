import pandas as pd
from itertools import product

class grid_search:
    def __init__(self, grid_path):
        self.grid_path = grid_path
    
    def read_grid(self):
        grid = pd.read_csv(self.grid_path)

        grid_cols = list(grid.columns)

        grid_combs = list(product(grid['batch size'], grid['epochs']))

        comb_dict_list = []
        for comb in grid_combs:
            comb = list(comb)

            comb_dict = dict(zip(grid_cols, comb))

            comb_dict_list.append(comb_dict)

        return comb_dict_list
