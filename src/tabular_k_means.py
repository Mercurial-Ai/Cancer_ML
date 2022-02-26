import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from tokenize_dataset import tokenize_dataset
from PeakCluster import PeakCluster
from collections import Counter

df = pd.read_csv("data\METABRIC_RNA_Mutation\METABRIC_RNA_Mutation.csv", low_memory=False)

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
