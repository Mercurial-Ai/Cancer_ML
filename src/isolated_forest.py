
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd 
import argparse 

def isolated_forest(features, target):

    isolated_forest=IsolationForest(n_estimators=100, n_jobs=-1,random_state=42) 

    isolated_forest.fit(features)
    predicted=isolated_forest.predict(features)

    target = np.expand_dims(target, -1)

    elements, counts = np.unique(predicted, return_counts=True)
    counts_dict = dict(zip(elements, counts))
    print(counts_dict)

    predicted = pd.DataFrame(predicted)
    predicted.to_csv('data_anomaly.csv')
