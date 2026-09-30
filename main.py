import numpy as np
import os
import json
from core.base import MetricRelation, BaseMetricIndex, QueryNode
from modules.predicate_search.engine import PredicateSearchModule
from modules.query_engine.parser import SQLParser
from modules.union_management.union import UnionManager
from modules.metric_indexing.indexing import MetricIndex
from modules.entity_linking.linker import EntityLinker
from data.data_generator import generate_synthetic_data

# Configurations - Can be modified by students for different scales
DATA_SIZE = 100
VECTOR_DIM = 128
DATA_DIR = "data"

class SimpleMetricRelation(MetricRelation):
    """
    A concrete implementation of MetricRelation for demonstration purposes.
    It loads data from the generated NumPy and JSON files.
    """
    def __init__(self, name, vectors, metadata):
        self.name = name
        self.vectors = vectors
        self.metadata = metadata
        self.ids = [m['id'] for m in metadata]

    def get_vectors(self, ids):
        """Retrieve vectors by looking up indices of given IDs."""
        indices = [self.ids.index(i) for i in ids]
        return self.vectors[indices]

    def get_metadata(self, ids):
        """Retrieve metadata dictionaries for given IDs."""
        indices = [self.ids.index(i) for i in ids]
        return [self.metadata[i] for i in indices]

def main():
    print("=== MetricDB System Integration Demo ===")
    
    # 1. Data Generation and Loading
    print(f"\n[Step 1] Preparing synthetic data (Size: {DATA_SIZE}, Dim: {VECTOR_DIM})...")
    
    # Always regenerate or check existence with correct parameters
    # In a real scenario, students might want to use different datasets
    vector_path = os.path.join(DATA_DIR, "vectors.npy")
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    
    # Generate data if not exists or if parameters changed (simple check)
    generate_new = True
    if os.path.exists(vector_path):
        existing_vectors = np.load(vector_path)
        if existing_vectors.shape == (DATA_SIZE, VECTOR_DIM):
            generate_new = False
            
    if generate_new:
        generate_synthetic_data(num_samples=DATA_SIZE, dimension=VECTOR_DIM, output_dir=DATA_DIR)
    
    vectors = np.load(vector_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    relation = SimpleMetricRelation("Inventory", vectors, metadata)
    print(f"Loaded relation '{relation.name}' with {len(metadata)} items.")

    # 2. Module Demonstration
    # This section shows how the independent modules are instantiated and wired together.
    print("\n[Step 2] Initializing Modules...")
    
    # Module 4: Metric Indexing
    # Students implementing Module 4 will work on the build() and search() logic.
    print("- Initializing Metric Index (Module 4)...")
    index = MetricIndex()
    index.build(relation)
    
    # Module 1: Predicate Search
    # Students implementing Module 1 will focus on combining 'index' search with 'relation' metadata filters.
    print("- Initializing Predicate Search (Module 1)...")
    ps_module = PredicateSearchModule(index, relation)
    
    # Module 2: Query Engine
    # Students implementing Module 2 will work on parsing SQL and optimizing the execution plan.
    print("- Initializing Query Engine (Module 2)...")
    parser = SQLParser()
    sql = "SELECT * FROM Inventory WHERE category = 'electronics' AND vector SIMILAR TO query"
    query_node = parser.parse(sql)
    
    # Module 3: Union Management
    # Students implementing Module 3 will handle operations across multiple MetricRelations.
    print("- Initializing Union Manager (Module 3)...")
    union_mgr = UnionManager()
    
    # Module 5: Entity Linking
    # Students implementing Module 5 will use geometric hashing to link different relations.
    print("- Initializing Entity Linker (Module 5)...")
    linker = EntityLinker()

    print("\n=== Integration Demo Completed ===")
    print("The cluster modules are initialized. See README.md for the dissertation indexing scope and remaining scaffold modules.")
    print(f"To run unit tests: python -m unittest discover tests")

if __name__ == "__main__":
    main()
