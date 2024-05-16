import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np


def cluster_fingerprints(data, labels):
    linked = linkage(data, method='single', metric='euclidean')
    dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=True)
    plt.show()

    clusters = fcluster(linked, 3, criterion='maxclust')
    return clusters

def to_matrix(data):
    return data.iloc[:, 1:].values

def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    data = load_data("./data/result_smile_descriptor.csv")
    print(data.head())
    labels = data.iloc[:, 0].values
    
    matrix = to_matrix(data)
    
    
    
    clusters = cluster_fingerprints(matrix, labels)
    data["Cluster"] = clusters
    print("========== Hasil Cluter ==========")
    sorted_data = data.sort_values(by=['Cluster'], ascending=True)
    print(sorted_data[['Name', 'Cluster']])

if __name__ == "__main__":
    main()
