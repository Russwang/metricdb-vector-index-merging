import unittest
import numpy as np
import os
import json
from main import SimpleMetricRelation
from modules.metric_indexing.indexing import MetricIndex

class TestMetricIndexing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use generated data or create mock
        data_dir = "data"
        vector_path = os.path.join(data_dir, "vectors.npy")
        metadata_path = os.path.join(data_dir, "metadata.json")
        
        if os.path.exists(vector_path) and os.path.exists(metadata_path):
            cls.vectors = np.load(vector_path)
            with open(metadata_path, "r") as f:
                cls.metadata = json.load(f)
            cls.relation = SimpleMetricRelation("TestRelation", cls.vectors, cls.metadata)
            cls.dim = cls.vectors.shape[1]
        else:
            cls.dim = 128
            cls.vectors = np.random.rand(10, cls.dim).astype(np.float32)
            cls.metadata = [{"id": f"item_{i}"} for i in range(10)]
            cls.relation = SimpleMetricRelation("MockRelation", cls.vectors, cls.metadata)

    def setUp(self):
        self.index = MetricIndex()

    def test_build_index(self):
        """Test if the index can be built from a MetricRelation."""
        try:
            self.index.build(self.relation)
        except Exception as e:
            self.fail(f"index.build() failed with {e}")

    def test_search_interface(self):
        """Test if the search method returns a list of results."""
        self.index.build(self.relation)
        query = np.random.rand(self.dim).astype(np.float32)
        results = self.index.search(query, k=5)
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()
