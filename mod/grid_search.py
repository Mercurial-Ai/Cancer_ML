import pandas as pd
from itertools import product
import math
import xlwt
from xlwt import Workbook

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

class write_excel:
    def __init__(self, sheet_path, hyp_dict_list):
        self.path = sheet_path
        self.hyps = hyp_dict_list

    def run(self):
        wb = Workbook()

        sheet = wb.add_sheet('Sheet 1')

        # add column labels to sheet
        i = 0
        perf_tuple = self.hyps[0]
        hyp_dict = perf_tuple[0]
        labels = list(hyp_dict.keys())
        for label in labels:
            sheet.write(0, i, label)

            i = i + 1

        sheet.write(0, i, 'accuracy')

        # add corresponding hyperparameters and performance data to each column
        i = 1
        for perf_tuple in self.hyps:
            hyp_dict = perf_tuple[0]
            accuracy = perf_tuple[-1]
            j = 0
            for label in labels:
                sheet.write(i, j, hyp_dict[label])

                j = j + 1

            sheet.write(i, j, accuracy)

            i = i + 1

        wb.save(self.path)
