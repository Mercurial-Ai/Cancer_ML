
# Assign ascending ints to y vals 1, 2, 3, ...
def tokenize_y(y):

    y = list(y)

    # remove duplicates to identify vals
    y = list(set(y))
    y.sort()

    i = 0
    binary_dict = {}
    for val in y:
        binary_dict[val] = i

        y[i] = i

        i = i + 1
    
    return y
    