from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def isolation_forest(features, target):

    isolated_forest=IsolationForest(n_estimators=10, n_jobs=-1, random_state=42) 

    if type(features) == tuple or type(features) == list:
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

    outlier_indices = []
    i = 0
    for prediction in predicted:
        if prediction == -1:
            outlier_indices.append(i)

        i = i + 1

    pca = PCA(n_components=3)

    if type(features) == tuple or type(features) == list:
        features = concatenated_array

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = pca.fit_transform(features)

    fig = plt.figure()
    fig.suptitle("3D PCA of Features with Outliers and Inliers")
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(features[:, 0], features[:, 1], zs=features[:, 2], s=4, lw=1, label="inliers", c="green")

    ax.scatter(features[outlier_indices,0], features[outlier_indices,1], features[outlier_indices,2], lw=2, s=60, marker='x', c='red', label='outliers')

    ax.legend()

    plt.savefig("3d_outlier_pca" + str(features.shape) + ".png")

    return predicted
