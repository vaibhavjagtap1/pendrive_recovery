"""
Face clusterer module – groups face embeddings into person identities.

Uses DBSCAN (preferred – no need to specify number of clusters) with
a fallback to K-Means when DBSCAN fails to produce useful clusters.

Each cluster corresponds to one person ("Person_1", "Person_2", …).
Faces that do not belong to any cluster are labelled "Unknown".
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Default DBSCAN parameters
# eps=0.5 works well for normalised cosine-distance embeddings
DEFAULT_EPS = 0.5
DEFAULT_MIN_SAMPLES = 2


class ClusterResult:
    """Holds the result of a clustering run."""

    def __init__(self, labels: np.ndarray, n_clusters: int, n_noise: int):
        self.labels = labels        # Array of cluster IDs (-1 = noise/unknown)
        self.n_clusters = n_clusters
        self.n_noise = n_noise

    def cluster_name(self, label: int) -> str:
        if label == -1:
            return "Unknown"
        return f"Person_{label + 1}"


class FaceClusterer:
    """
    Clusters face embedding vectors using DBSCAN.

    Parameters eps and min_samples can be tuned based on the quality of
    the embeddings and the expected number of faces per person.
    """

    def __init__(
        self,
        eps: float = DEFAULT_EPS,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        metric: str = "cosine",
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(self, embeddings: List[np.ndarray]) -> ClusterResult:
        """
        Cluster a list of face embeddings.

        Args:
            embeddings: List of 1-D numpy arrays (same dimensionality).

        Returns:
            ClusterResult with per-embedding cluster labels.
        """
        if not embeddings:
            return ClusterResult(np.array([], dtype=int), 0, 0)

        X = np.array(embeddings, dtype=np.float32)
        # L2-normalise so cosine distance = euclidean distance on unit sphere
        X = normalize(X, norm="l2")

        labels = self._run_dbscan(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        logger.info(
            "Clustering: %d embeddings -> %d clusters, %d noise points",
            len(embeddings), n_clusters, n_noise,
        )
        return ClusterResult(labels, n_clusters, n_noise)

    def assign_labels(
        self, embeddings: List[np.ndarray]
    ) -> List[str]:
        """
        Convenience method: returns string person labels for each embedding.

        Args:
            embeddings: List of face embedding vectors.

        Returns:
            List of strings like "Person_1", "Person_2", "Unknown".
        """
        result = self.cluster(embeddings)
        return [result.cluster_name(int(lbl)) for lbl in result.labels]

    def best_representative(
        self,
        embeddings: List[np.ndarray],
        labels: np.ndarray,
        cluster_id: int,
    ) -> Optional[int]:
        """
        Find the index of the best (closest to centroid) embedding
        for a given cluster.

        Args:
            embeddings: List of all embeddings.
            labels: Cluster label array from ClusterResult.
            cluster_id: The cluster ID to find the representative for.

        Returns:
            Index into embeddings, or None if cluster not found.
        """
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not indices:
            return None
        cluster_embs = np.array([embeddings[i] for i in indices], dtype=np.float32)
        centroid = cluster_embs.mean(axis=0)
        distances = np.linalg.norm(cluster_embs - centroid, axis=1)
        best_local = int(np.argmin(distances))
        return indices[best_local]

    def tune_eps(
        self, embeddings: List[np.ndarray], target_min_clusters: int = 2
    ) -> float:
        """
        Simple heuristic to find an eps value that produces at least
        target_min_clusters clusters.

        Useful when default eps creates too few or too many clusters.
        """
        if len(embeddings) < 2:
            return self.eps

        X = normalize(np.array(embeddings, dtype=np.float32), norm="l2")
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(self.min_samples, len(X) - 1), metric=self.metric)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_distances = np.sort(distances[:, -1])

        # Look for the "elbow" – where the sorted k-distance curve bends
        diffs = np.diff(k_distances)
        elbow = int(np.argmax(diffs))
        best_eps = float(k_distances[elbow])
        logger.debug("Auto-tuned eps: %.4f (was %.4f)", best_eps, self.eps)
        return best_eps

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_dbscan(self, X: np.ndarray) -> np.ndarray:
        """Run DBSCAN and return cluster labels."""
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        labels = dbscan.fit_predict(X)
        return labels
