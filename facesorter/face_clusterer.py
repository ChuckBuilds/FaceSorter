import numpy as np
from sklearn.cluster import DBSCAN

UNSORTED_LABEL = -1


class FaceClusterer:
    """
    Groups face embeddings into per-person clusters with DBSCAN over cosine
    distance. Embeddings are L2-normalized ArcFace vectors, so cosine
    distance is the natural metric (0 = identical, ~1 = unrelated).
    """

    def __init__(self, eps=0.5, min_samples=2):
        """
        Args:
            eps (float): Max cosine distance between two faces to be linked.
                         Lower = stricter (splits people), higher = looser
                         (merges people).
            min_samples (int): Minimum neighborhood size for a core point.
                         With 2+, one-off false detections land in the
                         "unsorted" bucket (label -1) instead of each
                         becoming their own person folder.
        """
        self.eps = eps
        self.min_samples = min_samples

    def cluster_faces(self, encodings):
        """
        Clusters a list/array of embeddings.

        Returns:
            (labels, num_clusters): labels is an int array aligned with the
            input; UNSORTED_LABEL (-1) marks noise points. num_clusters
            excludes the noise bucket.
        """
        if len(encodings) == 0:
            return np.array([], dtype=int), 0

        encodings_array = np.asarray(encodings)
        labels = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="cosine"
        ).fit(encodings_array).labels_
        num_clusters = len(set(labels) - {UNSORTED_LABEL})
        return labels, num_clusters
