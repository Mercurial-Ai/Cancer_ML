from xlwt import Workbook

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
            accuracy = perf_tuple[-1][0]
            j = 0
            for label in labels:
                sheet.write(i, j, hyp_dict[label])

                j = j + 1

            sheet.write(i, j, accuracy)

            i = i + 1

        wb.save(self.path)
