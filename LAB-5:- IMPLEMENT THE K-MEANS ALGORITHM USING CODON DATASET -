import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load and prepare the Iris dataset
features = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
cluster_labels = kmeans.labels_

# Print cluster centers and indices of closest points
print("Cluster Centers:\n", kmeans.cluster_centers_)
distances = np.linalg.norm(features.values[:, np.newaxis] - kmeans.cluster_centers_, axis=2)
print("\nIndices of Closest Points to Cluster Centers:\n", np.argmin(distances, axis=0))

# Plot clusters
plt.figure(figsize=(10, 7))
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
