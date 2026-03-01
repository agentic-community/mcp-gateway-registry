"""
Mock FAISS implementation for testing.

This module provides mock implementations of FAISS classes to avoid
loading the actual FAISS library during tests.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MockFaissIndex:
    """
    Mock implementation of FAISS index for testing.

    This mock simulates FAISS index behavior without requiring the actual
    FAISS library to be loaded.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize mock FAISS index.

        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
        self._vectors: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        logger.debug(f"Created MockFaissIndex with dimension {dimension}")

    @property
    def d(self) -> int:
        """Get the dimension of the index."""
        return self.dimension

    @property
    def ntotal(self) -> int:
        """Get the total number of vectors in the index."""
        return len(self._vectors)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Add vectors with specific IDs to the index.

        Args:
            vectors: Array of vectors to add (shape: [n, d])
            ids: Array of IDs for the vectors (shape: [n])
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )

        for i, vector_id in enumerate(ids):
            self._vectors[int(vector_id)] = vectors[i]

        logger.debug(f"Added {len(ids)} vectors to mock index (total: {self.ntotal})")

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index with auto-generated IDs.

        Args:
            vectors: Array of vectors to add (shape: [n, d])
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )

        n = vectors.shape[0]
        ids = np.arange(self._next_id, self._next_id + n)
        self.add_with_ids(vectors, ids)
        self._next_id += n

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            query_vectors: Query vectors (shape: [n, d])
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices) arrays
        """
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vectors.shape[1]} does not match index dimension {self.dimension}"
            )

        n_queries = query_vectors.shape[0]
        n_vectors = self.ntotal

        if n_vectors == 0:
            # No vectors in index, return empty results
            distances = np.full((n_queries, k), float("inf"), dtype=np.float32)
            indices = np.full((n_queries, k), -1, dtype=np.int64)
            return distances, indices

        # Calculate distances for all vectors
        all_ids = np.array(list(self._vectors.keys()), dtype=np.int64)
        all_vectors = np.array([self._vectors[vid] for vid in all_ids])

        distances_list = []
        indices_list = []

        for query_vector in query_vectors:
            # Calculate L2 distances
            dists = np.linalg.norm(all_vectors - query_vector, axis=1)

            # Get top k
            k_actual = min(k, len(dists))
            top_k_indices = np.argsort(dists)[:k_actual]

            # Build result arrays
            result_distances = np.full(k, float("inf"), dtype=np.float32)
            result_indices = np.full(k, -1, dtype=np.int64)

            result_distances[:k_actual] = dists[top_k_indices]
            result_indices[:k_actual] = all_ids[top_k_indices]

            distances_list.append(result_distances)
            indices_list.append(result_indices)

        distances = np.array(distances_list)
        indices = np.array(indices_list)

        logger.debug(f"Searched {n_queries} queries, found {k} neighbors each")
        return distances, indices

    def remove_ids(self, ids: np.ndarray) -> int:
        """
        Remove vectors with specific IDs from the index.

        Args:
            ids: Array of IDs to remove

        Returns:
            Number of vectors removed
        """
        removed = 0
        for vector_id in ids:
            if int(vector_id) in self._vectors:
                del self._vectors[int(vector_id)]
                removed += 1

        logger.debug(f"Removed {removed} vectors from mock index (remaining: {self.ntotal})")
        return removed

    def reset(self) -> None:
        """Reset the index to empty state."""
        self._vectors.clear()
        self._next_id = 0
        logger.debug("Reset mock index")


class MockIndexIDMap:
    """
    Mock implementation of FAISS IndexIDMap wrapper.

    This wraps a MockFaissIndex to provide ID mapping functionality.
    """

    def __init__(self, index: MockFaissIndex):
        """
        Initialize mock IndexIDMap.

        Args:
            index: Underlying mock index
        """
        self.index = index
        logger.debug("Created MockIndexIDMap")

    @property
    def d(self) -> int:
        """Get the dimension of the index."""
        return self.index.d

    @property
    def ntotal(self) -> int:
        """Get the total number of vectors."""
        return self.index.ntotal

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors with IDs."""
        self.index.add_with_ids(vectors, ids)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors."""
        return self.index.search(query_vectors, k)

    def remove_ids(self, ids: np.ndarray) -> int:
        """Remove vectors by IDs."""
        return self.index.remove_ids(ids)

    def reset(self) -> None:
        """Reset the index."""
        self.index.reset()


def create_mock_faiss_module() -> Any:
    """
    Create a mock FAISS module for testing.

    This returns a module-like object that can be used to replace
    the faiss import in tests.

    Returns:
        Mock FAISS module object
    """

    class MockFaissModule:
        """Mock FAISS module."""

        @staticmethod
        def IndexFlatL2(d: int) -> MockFaissIndex:
            """Create a flat L2 index."""
            logger.debug(f"Creating MockFaissIndex with dimension {d}")
            return MockFaissIndex(d)

        @staticmethod
        def IndexFlatIP(d: int) -> MockFaissIndex:
            """Create a flat Inner Product index (for cosine similarity)."""
            logger.debug(f"Creating MockFaissIndex (IP) with dimension {d}")
            return MockFaissIndex(d)

        @staticmethod
        def IndexIDMap(index: MockFaissIndex) -> MockIndexIDMap:
            """Create an ID map wrapper."""
            logger.debug("Creating MockIndexIDMap")
            return MockIndexIDMap(index)

        @staticmethod
        def read_index(filepath: str) -> MockFaissIndex:
            """
            Mock read_index that returns an empty index.

            In real tests, the index will be populated separately.
            """
            logger.debug(f"Mock reading FAISS index from {filepath}")
            return MockFaissIndex()

        @staticmethod
        def write_index(index: MockFaissIndex, filepath: str) -> None:
            """Mock write_index that does nothing."""
            logger.debug(f"Mock writing FAISS index to {filepath}")

    return MockFaissModule()
