from core.base import MetricRelation, BaseMetricIndex
from typing import List, Dict, Any
import numpy as np

class PredicateSearchModule:
    """
    Module 1: Predicate Vector Search (Assigned to: Student 1)
    
    This module is responsible for hybrid search queries that combine relational 
    filtering (predicates) with vector similarity search.
    
    Task for Student 1:
    Implement 'hybrid_search' which efficiently combines vector results from the index
    and attribute results from the relation. Consider different strategies:
    1. Pre-filtering: Filter metadata first, then search the index among survivors.
    2. Post-filtering: Search index first, then filter results by metadata.
    """
    def __init__(self, index: BaseMetricIndex, relation: MetricRelation):
        """
        Initialize with a metric index and the underlying relation.
        
        Args:
            index: An instance of BaseMetricIndex (Module 4).
            relation: An instance of MetricRelation containing vectors and metadata.
        """
        self.index = index
        self.relation = relation

    def hybrid_search(self, query_vector: np.ndarray, filters: Dict[str, Any], k: int = 10) -> List[str]:
        """
        Perform a hybrid search.
        
        Args:
            query_vector: The query embedding.
            filters: A dictionary of predicates (e.g., {"category": "electronics"}).
            k: Number of top results to return.
            
        Returns:
            A list of item IDs that satisfy the filters and are most similar to the query.
        """
        # TODO: Implement pre-filtering or post-filtering logic.
        # Example pseudo-code for post-filtering:
        # 1. candidate_ids = self.index.search(query_vector, k=k*10)
        # 2. metadata = self.relation.get_metadata(candidate_ids)
        # 3. filtered_ids = [id for id, meta in zip(candidate_ids, metadata) if matches(meta, filters)]
        # 4. return filtered_ids[:k]
        return []
