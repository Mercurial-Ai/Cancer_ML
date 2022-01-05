from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import pandas as pd 
import numpy as np
import math

def isolation_forest(features, target):

    isolated_forest=IsolationForest(n_estimators=100, n_jobs=-1, random_state=42, max_depth=30) 

    if type(features) == tuple:
        # concatenate features for image clinical
        clinical_array = features[0]
        image_array = features[1]

        new_array = np.empty(shape=(image_array.shape[0], int(image_array.shape[1])**2))
        i = 0
        for image in image_array:
            image = np.reshape(image, (1, int(image_array.shape[1])**2))
            new_array[i] = image

            i = i + 1
        
        image_array = new_array

        concatenated_array = np.concatenate((clinical_array, image_array), axis=1)
        
        isolated_forest.fit(concatenated_array, target)
        predicted = isolated_forest.predict(concatenated_array)
    else:
        isolated_forest.fit(features, target)
        predicted = isolated_forest.predict(features)

    predicted_df = pd.DataFrame(predicted)
    predicted_df.to_csv('data_anomaly.csv')

    return predicted
