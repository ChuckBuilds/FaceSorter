from sklearn.cluster import DBSCAN
import numpy as np
from itertools import combinations

class FaceClusterer:
    """
    Groups face encodings into clusters using the DBSCAN algorithm.
    """

    def __init__(self, eps=0.5, min_samples=1):
        """
        Initializes the FaceClusterer.

        Args:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
                               With min_samples=1, no points are considered noise.
        """
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")

    def cluster_faces(self, encodings):
        """
        Performs DBSCAN clustering on a list of face encodings.

        Args:
            encodings (list): A list of 128-dimensional face encodings.

        Returns:
            A tuple containing:
            - cluster_labels (np.array): An array where each element is the cluster ID for the corresponding encoding.
            - num_clusters (int): The total number of unique clusters found.
        """
        if not encodings:
            return np.array([]), 0

        # Ensure encodings are in a format scikit-learn can use (list of lists or 2D array)
        if isinstance(encodings[0], np.ndarray):
            encodings_array = np.array(encodings)
        else:
            encodings_array = encodings

        self.dbscan.fit(encodings_array)
        cluster_labels = self.dbscan.labels_

        # The total number of clusters is the number of unique labels.
        # Since min_samples=1, there will be no noise points (-1).
        num_clusters = len(set(cluster_labels)) if len(cluster_labels) > 0 else 0
        
        return cluster_labels, num_clusters

    @staticmethod
    def get_cluster_centroids(encodings, labels):
        """
        Calculates the centroid (average encoding) for each cluster.

        Args:
            encodings (np.array): The array of all face encodings.
            labels (np.array): The cluster label for each encoding.

        Returns:
            dict: A dictionary mapping each cluster ID to its centroid vector.
        """
        centroids = {}
        unique_labels = set(labels)
        for label in unique_labels:
            # Get all encodings belonging to the current cluster
            cluster_encodings = encodings[labels == label]
            # Calculate the mean of those encodings
            centroids[label] = np.mean(cluster_encodings, axis=0)
        return centroids

    @staticmethod
    def find_merge_candidates(centroids, threshold=0.6):
        """
        Finds pairs of clusters that are potential candidates for merging based on centroid distance.

        Args:
            centroids (dict): A map of cluster IDs to their centroid vectors.
            threshold (float): The distance threshold. Pairs with a centroid distance below this value will be suggested for merging.

        Returns:
            list: A list of tuples, where each tuple contains two cluster IDs that are candidates for merging.
        """
        candidates = []
        # Get all unique pairs of cluster IDs
        cluster_ids = list(centroids.keys())
        if len(cluster_ids) < 2:
            return []

        for id1, id2 in combinations(cluster_ids, 2):
            centroid1 = centroids[id1]
            centroid2 = centroids[id2]
            
            # Calculate the Euclidean distance between the two centroids
            distance = np.linalg.norm(centroid1 - centroid2)
            
            if distance < threshold:
                # Add the pair (in a consistent order) to the candidates list
                candidates.append(tuple(sorted((id1, id2))))
        
        return candidates 