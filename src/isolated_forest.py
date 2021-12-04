
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

from src.get_distribution import get_distribution

def isolated_forest(features, target):

    isolated_forest=IsolationForest(n_estimators=100, n_jobs=-1,random_state=42) 

    predicted = isolated_forest.fit_predict(features, target)

    predicted_df = pd.DataFrame(predicted)
    predicted_df.to_csv('data_anomaly.csv')

    # get distribution of outlier data (-1 is outlier, 1 is inlier)
    predicted_dist = get_distribution(predicted)

    return predicted_dist

