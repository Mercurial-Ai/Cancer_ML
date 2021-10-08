from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.ops.gen_array_ops import empty

def tokenize_dataset(pd_dataset):

    # replace missing spots with a string then after dataset is encoded, replace with mean of column
    df = pd_dataset.fillna('empty')

    if df.shape[0] >= df.shape[1]:
        long_axis = df.shape[0]
        short_axis = df.shape[1]
    else:
        long_axis = df.shape[1]
        short_axis = df.shape[0]

    word_list = []
    for i in range(long_axis):
        for n in range(short_axis):
            
            if long_axis == df.shape[0]:
                data = df.iloc[i, n]
            else:
                data = df.iloc[n, i]

            if str(type(data)) == "<class 'str'>":

                # list of chars to be removed from data
                char_blocked = [' ', '.', '/', '-', '_', '>', '+', ',', ')', '(', '*',
                                '=', '?', ':', '[', ']', '#', '!', '\n', '\\', '}',
                                '{', ';', '%', '"']
                
                for char in char_blocked:
                    if char in data:
                        data = data.replace(char, '')

                data = data.lower()

                if long_axis == df.shape[0]:
                    df.iloc[i, n] = data
                else:
                    df.iloc[n, i] = data

                word_list.append(data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)
    code_dict = tokenizer.word_index

    for i in range(long_axis):
        for n in range(short_axis):
            if long_axis == df.shape[0]:
                data = df.iloc[i, n]
            else: 
                data = df.iloc[n, i]

            if str(type(data)) == "<class 'str'>":
                
                data = int(code_dict[data])

            if long_axis == df.shape[0]: 
                df.iloc[i, n] = data
            else: 
                df.iloc[n, i] = data

    # replace spots previously denoted as 'empty' with mean of column
    for column in list(df.columns):
        col = df[column].copy()

        empty_indices = []

        i = 0
        for val in col:
            if val == code_dict['empty']:
                empty_indices.append(i)

            i = i + 1

        # get series labels at the empty indices for .drop function
        col_labels = list(col.index)

        empty_labels = []
        for index in empty_indices:
            empty_labels.append(col_labels[index])

        col_without_empty = col.drop(empty_labels)

        col_mean = col_without_empty.mean()

        i = 0
        for val in col:
            if val == code_dict['empty']:
                col[i] = col_mean

            i = i + 1
        
        df[column] = col

    # convert all cols to numeric vals
    df = df.astype('int64')

    return df
    