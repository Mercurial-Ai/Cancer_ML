import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from get_distribution import get_distribution
from tokenize_dataset import tokenize_dataset
from PeakCluster import PeakCluster
from collections import Counter

df = pd.read_csv("data\Duke-Breast-Cancer-MRI\Clinical and Other Features (edited).csv", low_memory=False)

column_means = df.mean()
df = df.fillna(column_means)

df = tokenize_dataset(df)

scaler = MinMaxScaler()
data = scaler.fit_transform(df)

model = PeakCluster(data)
label_counts = dict(Counter(model.labels_))

n_clusters = len(list(set(model.labels_)))

neigh = KNeighborsClassifier(n_neighbors=n_clusters)
neigh.fit(data, model.labels_)

inferences = list(neigh.predict(data))
inference_distribution = get_distribution(inferences)

# equalize classes identified
classes_equal = False
indices_removed = []
while not classes_equal:
    most_frequent_class = max(set(inferences), key=inferences.count)
    most_frequent_count = max(inference_distribution.values())*len(inferences)*0.01
    least_frequent_count = min(inference_distribution.values())*len(inferences)*0.01
    num_to_remove = int(round((most_frequent_count - least_frequent_count), 0))

    for i in range(num_to_remove):
        indices_removed.append(inferences.index(most_frequent_class))
        data = np.delete(data, inferences.index(most_frequent_class), axis=0)
        inferences.remove(most_frequent_class)

    inference_distribution = get_distribution(inferences)

    # if the distribution of all classes are equal
    if all(element == list(inference_distribution.values())[0] for element in list(inference_distribution.values())):
        classes_equal = True
