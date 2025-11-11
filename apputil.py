
# Exercise 1:

import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Create a KMeans clustering model using scikit-learn
    """
    model = KMeans(n_clusters=k, random_state=42)
    # Fit the model
    model.fit(X)
    # Extract centroids and labels
    centroids = model.cluster_centers_
    labels = model.labels_
    # Return as a tuple
    return centroids, labels

# Example
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

centroids, labels = kmeans(X, k=3)

# Print Results
print("Centroids:\n", centroids)
print("Labels:", labels)
print("Centroid for 2nd cluster:", centroids[1])
print("Cluster assignment for 3rd data point:", labels[2])

# Exercise 2:

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# Load the diamonds dataset
diamonds = sns.load_dataset('diamonds')
numeric_diamonds = diamonds.select_dtypes(include=[np.number])

# kmeans function from Exercise 1
def kmeans(X, k):
    """
    Create a KMeans clustering model using scikit-learn
    """
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_
    return centroids, labels

# Define the kmeans_diamonds function
def kmeans_diamonds(n, k):
    """
    Runs k-means clustering on the diamonds data.
    """
    # Select the first n rows of numeric columns
    X = numeric_diamonds.head(n).to_numpy()
    
    # kmeans function from Exercise 1
    centroids, labels = kmeans(X, k)
    
    return centroids, labels


# Example
centroids, labels = kmeans_diamonds(n=1000, k=5)

# Print Results
print("Centroids:\n", centroids)
print("Labels:", labels[:15])
print("Centroid for 4th cluster:", centroids[3])
print("Cluster assignment for 10th diamond:", labels[9])


# Exercise 3:

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

# (From Exercise 2)
diamonds = sns.load_dataset('diamonds')
numeric_diamonds = diamonds.select_dtypes(include=[np.number])

def kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    return model.cluster_centers_, model.labels_

def kmeans_diamonds(n, k):
    X = numeric_diamonds.head(n).to_numpy()
    centroids, labels = kmeans(X, k)
    return centroids, labels

# (Exercise 3)
def kmeans_timer(n, k, n_iter=5):
    """
    Runs kmeans_diamonds(n, k) exactly n_iter times,
    records the runtime for each run, and returns the average time in seconds.
    """
    times = []

    for i in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        runtime = time() - start
        times.append(runtime)
        print(f"Run {i+1}/{n_iter}: {runtime:.4f} seconds")

    avg_time = np.mean(times)
    return avg_time

# Example
avg_runtime = kmeans_timer(n=1000, k=5, n_iter=5)
print(f"\nAverage runtime over 5 iterations: {avg_runtime:.4f} seconds")