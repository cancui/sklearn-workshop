from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset, which has 3 dimensions of features
iris = load_iris()

# Use PCA to reduce Iris' 3D data into 2D data that best preserves the original data's variance.
# One way that PCA is useful for reducing higher-dimensional data into 2D/3D for visualization.
pca = PCA(n_components=2)
iris_data_2d = pca.fit_transform(iris.data)




####################### 2 Clusters #######################

# Create a k-means object that clusters data into 2 clusters
kmeans = KMeans(n_clusters=2)

# Sort each datapoint into one cluster
cluster_of_data = kmeans.fit_predict(iris_data_2d)
cluster1 = iris_data_2d[np.where(cluster_of_data == 0)]
cluster2 = iris_data_2d[np.where(cluster_of_data == 1)]

# Plot clusters
plt.figure()
plt.scatter(cluster1[:,0], cluster1[:,1], color='red')
plt.scatter(cluster2[:,0], cluster2[:,1], color='blue')
plt.xlabel('Feature A')
plt.ylabel('Feature B')



####################### 4 Clusters #######################

# Create a k-means object that clusters data into 4 clusters, get the cluster of each datapoint,
# and separate the data into the 4 clusters
kmeans = KMeans(n_clusters=4)
cluster_of_data = kmeans.fit_predict(iris_data_2d)
clusters = [iris_data_2d[np.where(cluster_of_data == i)] for i in range(4)]

# Plot the 4 clusters
plt.figure()
for cluster in clusters:
    plt.scatter(cluster[:,0], cluster[:,1])
plt.xlabel('Feature A')
plt.ylabel('Feature B')

# Get the cluster of a new datapoint
new_data = np.asarray([[1,-1.5]])
new_data_cluster = kmeans.predict(new_data)[0]
print('New datapoint {} falls in cluster {}'.format(new_data, new_data_cluster))

# Plot unclustered data
plt.figure()
plt.scatter(iris_data_2d[:,0], iris_data_2d[:,1])
plt.xlabel('Feature A')
plt.ylabel('Feature B')
plt.show()