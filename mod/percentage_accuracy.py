def percentageAccuracy(iterable1, iterable2):

        def roundList(iterable):

            if str(type(iterable)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
                iterable = iterable.numpy()
            roundVals = []
            if int(iterable.ndim) == 1:
                for i in iterable:
                    i = round(i, 0)
                    roundVals.append(i)

            elif int(iterable.ndim) == 2:
                for arr in iterable:
                    for i in arr:
                        i = round(i, 0)
                        roundVals.append(i)

            elif int(iterable.ndim) == 3:
                for dim in iterable:
                    for arr in dim:
                        for i in arr:
                            i = round(i, 0)
                            roundVals.append(i)

            elif int(iterable.ndim) == 4:
                for d in iterable:
                    for dim in d:
                        for arr in dim:
                            for i in arr:
                                i = round(i, 0)
                                roundVals.append(i)

            else:
                print("Too many dimensions--ERROR")

            return roundVals

        rounded1 = roundList(iterable1)
        rounded2 = roundList(iterable2)

        # remove negative zeros from lists
        i = 0
        for vals in rounded1:
            if int(vals) == -0 or int(vals) == 0:
                vals = abs(vals)
                rounded1[i] = vals

            i = i + 1

        i = 0
        for vals in rounded2:
            if int(vals) == -0 or int(vals) == 0:
                vals = abs(vals)
                rounded2[i] = vals

            i = i + 1

        numCorrect = len([i for i, j in zip(rounded1, rounded2) if i == j])

        listLen = len(rounded1)

        percentCorr = numCorrect / listLen
        percentCorr = percentCorr * 100

        percentCorr = round(percentCorr, 2)

        return percentCorr