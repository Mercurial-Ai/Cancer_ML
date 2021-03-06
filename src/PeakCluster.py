import numpy as np  
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score 
from scipy.interpolate import UnivariateSpline
import math

def PeakCluster(data):

    data_shape = data.shape

    clusters=range(3,20) 

    summed_square_distance=[]
    calinski_score=[]

    print(np.unique(data))
    for i in clusters: 
        kmeans=KMeans(n_clusters=i,init='random',random_state=1)
        kmeans=kmeans.fit(data)
        summed_square_distance.append(kmeans.inertia_)
        calinski_score.append(calinski_harabasz_score(data,kmeans.labels_))

    #2nd derivative of elbow curve to find optimal number of clusters 
    spline    = UnivariateSpline(clusters,summed_square_distance)
    spline_d2 = spline.derivative(n=2) 

    d2_list = list(spline_d2(clusters))
    idx_max = max(range(len(d2_list)),key=d2_list.__getitem__)
    n_clusters=idx_max + min(clusters)

    plt.figure() 
    plt.plot(clusters,spline_d2(clusters))
    plt.savefig('d2_ssd.png') 

    print(n_clusters)
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(data)
    Y = model.labels_ 

    #PCA to 2D 
    pca           = PCA(n_components=2)
    pca_transform = pca.fit_transform(data)

    plt.figure()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    i=0
    pca_holder = pd.DataFrame(columns=['pca0','pca1','label'])
    pca_holder['pca0']   = pca_transform[:,0]
    pca_holder['pca1']   = pca_transform[:,1]
    pca_holder['labels'] = Y 
    unique_labels = set(pca_holder['labels']) 
    plt.scatter(pca_transform[:,0],pca_transform[:,1], c=Y,cmap=plt.cm.jet)
    plt.tight_layout() 
    plt.savefig('pca_plot.png')

    def infer_cluster_labels(kmeans, actual_labels):
        inferred_labels = {}
        for i in range(kmeans.n_clusters):

            # find index of points in cluster
            labels = []
            index = np.where(kmeans.labels_ == i)

            # append actual labels for each point in cluster
            labels.append(actual_labels[index])

            # determine most common label
            if len(labels[0]) == 1:
                counts = np.bincount(labels[0])
            else:
                counts = np.bincount(np.squeeze(labels))
            # assign the cluster to a value in the inferred_labels dictionary
            if np.argmax(counts) in inferred_labels:
                # append the new number to the existing array at this slot
                inferred_labels[np.argmax(counts)].append(i)
            else:
                # create a new array in this slot
                inferred_labels[np.argmax(counts)] = [i]
        
        return inferred_labels

    centroids = model.cluster_centers_

    # check if data is imagery or tabular
    if len(data_shape) == 3:
        images = centroids.reshape(n_clusters, int(math.sqrt(centroids.shape[1])), int(math.sqrt(centroids.shape[1])))
        images *= 255
        images = images.astype(np.uint8)

    # determine cluster labels
    cluster_labels = infer_cluster_labels(model, Y)

    if len(data_shape) == 3:
        # create figure with subplots using matplotlib.pyplot
        fig, axs = plt.subplots(int(n_clusters/int(np.sqrt(n_clusters))), int(np.sqrt(n_clusters)), figsize = (20, 20))
        plt.gray()

        # loop through subplots and add centroid images
        for i, ax in enumerate(axs.flat):
            
            # determine inferred label using cluster_labels dictionary
            for key, value in cluster_labels.items():
                if i in value:
                    ax.set_title('Inferred Label: {}'.format(key))
            
            # add image to subplot
            ax.matshow(images[i])
            ax.axis('off')
            
        # display the figure
        fig.savefig("centroid_img.png")

    return model
    