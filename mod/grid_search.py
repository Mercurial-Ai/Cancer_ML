
# dict of all hyperparamaters to be optimized with their type (numerical or categorical)
# if categorical, then a list will be inserted with ['categorical', list of classes]
# if numerical, then a list will be inserted with ['numerical', (tuple with range of classes)]
hyps = {'learning rate': ['numerical', (0.01, 0.1)], 'batch_size': ['numerical', (8, 128)], 
        'epochs': ['numerical', (5, 30)], 'loss function': ['categorical', ['adam', 'mean_squared_error', 'mean_absolute_error']], 
        'optimizer': ['categorical', []]}

