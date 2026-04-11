"""Tests for the AI face analysis modules."""

import numpy as np
import pytest

from src.ai.face_clusterer import FaceClusterer, ClusterResult


# -----------------------------------------------------------------------
# FaceClusterer
# -----------------------------------------------------------------------

class TestFaceClusterer:
    def _make_embeddings(self, n_clusters: int, n_per_cluster: int, dim: int = 128) -> list:
        """Synthesize embeddings: tight groups around random centroids."""
        rng = np.random.default_rng(42)
        embeddings = []
        for _ in range(n_clusters):
            centroid = rng.standard_normal(dim)
            centroid /= np.linalg.norm(centroid)
            for _ in range(n_per_cluster):
                noise = rng.standard_normal(dim) * 0.05
                emb = centroid + noise
                emb /= np.linalg.norm(emb)
                embeddings.append(emb)
        return embeddings

    def test_cluster_empty_input(self):
        clusterer = FaceClusterer()
        result = clusterer.cluster([])
        assert result.n_clusters == 0
        assert result.n_noise == 0
        assert len(result.labels) == 0

    def test_cluster_single_embedding(self):
        clusterer = FaceClusterer(min_samples=1)
        emb = np.ones(128) / np.sqrt(128)
        result = clusterer.cluster([emb])
        # With min_samples=1, single point should form a cluster
        assert len(result.labels) == 1

    def test_cluster_identifies_groups(self):
        clusterer = FaceClusterer(eps=0.3, min_samples=2)
        embeddings = self._make_embeddings(n_clusters=3, n_per_cluster=5)
        result = clusterer.cluster(embeddings)
        # Should find at least 2 clusters (noise may absorb one)
        assert result.n_clusters >= 2

    def test_assign_labels_returns_strings(self):
        clusterer = FaceClusterer(eps=0.3, min_samples=2)
        embeddings = self._make_embeddings(n_clusters=2, n_per_cluster=4)
        labels = clusterer.assign_labels(embeddings)
        assert len(labels) == len(embeddings)
        for lbl in labels:
            assert isinstance(lbl, str)
            assert lbl.startswith("Person_") or lbl == "Unknown"

    def test_cluster_name_unknown(self):
        result = ClusterResult(np.array([-1, 0, 1]), 2, 1)
        assert result.cluster_name(-1) == "Unknown"
        assert result.cluster_name(0) == "Person_1"
        assert result.cluster_name(1) == "Person_2"

    def test_best_representative(self):
        clusterer = FaceClusterer(eps=0.3, min_samples=2)
        embeddings = self._make_embeddings(n_clusters=2, n_per_cluster=5)
        result = clusterer.cluster(embeddings)
        # Find first non-noise cluster
        clusters = set(result.labels) - {-1}
        if clusters:
            cid = next(iter(clusters))
            idx = clusterer.best_representative(embeddings, result.labels, cid)
            assert idx is not None
            assert 0 <= idx < len(embeddings)

    def test_best_representative_missing_cluster(self):
        clusterer = FaceClusterer()
        embeddings = [np.ones(128)]
        labels = np.array([0])
        idx = clusterer.best_representative(embeddings, labels, 99)
        assert idx is None

    def test_tune_eps_returns_float(self):
        clusterer = FaceClusterer()
        embeddings = self._make_embeddings(n_clusters=2, n_per_cluster=5)
        eps = clusterer.tune_eps(embeddings)
        assert isinstance(eps, float)
        assert eps > 0

    def test_tune_eps_single_embedding(self):
        clusterer = FaceClusterer()
        emb = [np.ones(128) / np.sqrt(128)]
        eps = clusterer.tune_eps(emb)
        assert eps == clusterer.eps  # Falls back to default

    def test_clustering_accuracy_two_people(self):
        """Accuracy test: 2 clear clusters, expect >= 80% correct assignment."""
        n_per_cluster = 20
        clusterer = FaceClusterer(eps=0.3, min_samples=2)
        embeddings = self._make_embeddings(n_clusters=2, n_per_cluster=n_per_cluster)
        ground_truth = [0] * n_per_cluster + [1] * n_per_cluster
        result = clusterer.cluster(embeddings)

        # Map cluster IDs to ground truth classes via majority vote
        from collections import Counter
        total = len(embeddings)
        correct = 0
        for cid in set(result.labels):
            if cid == -1:
                continue
            indices = [i for i, l in enumerate(result.labels) if l == cid]
            gt_labels = [ground_truth[i] for i in indices]
            majority = Counter(gt_labels).most_common(1)[0][0]
            correct += sum(1 for g in gt_labels if g == majority)

        accuracy = correct / total
        assert accuracy >= 0.80, f"Clustering accuracy {accuracy:.2%} below threshold"
