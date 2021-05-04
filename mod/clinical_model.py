
class clinical:
    def __init__(self, data_file, target_vars, epochs_num):
        self.data_file = data_file
        self.target_vars = target_vars
        self.epochs_num = epochs_num

    def pre(self):

        # initialize bool as false
        multiple_targets = False

        if str(type(self.target_vars)) == "<class 'list'>" and len(self.target_vars) > 1:
            multiple_targets = True