from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class VectorSpace(ABC):
    """
    Describes a specific embedding space, defined by a model and its output dimensionality.
    MetricDB aims to decouple logical data from these physical spaces.
    """
    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

class MetricRelation(ABC):
    """
    The fundamental data structure in MetricDB. 
    A MetricRelation combines relational attributes (metadata) with metric attributes (vectors).
    Students should implement this interface to handle data loading and retrieval.
    """
    @abstractmethod
    def get_vectors(self, ids: List[str]) -> np.ndarray:
        """Retrieve vectors for a given list of identifiers."""
        pass

    @abstractmethod
    def get_metadata(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve metadata dictionaries for a given list of identifiers."""
        pass

class BaseMetricIndex(ABC):
    """
    Abstract interface for metric indexing structures.
    Unlike traditional vector indices, metric indices might handle multiple spaces 
    or be independent of specific vector coordinates.
    """
    @abstractmethod
    def build(self, data: MetricRelation):
        """Construct the index from a MetricRelation."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[str]:
        """Perform a k-nearest neighbor search and return item IDs."""
        pass

class QueryNode(ABC):
    """
    Base class for nodes in a logical query plan.
    Module 2 students will extend this to represent SQL operations like Selection, Projection, and Metric Join.
    """
    pass

class MetricQueryProcessor(ABC):
    """
    Interface for the query execution engine.
    It takes a logical query plan (QueryNode tree) and produces results.
    """
    @abstractmethod
    def execute(self, query: QueryNode) -> List[Any]:
        """Execute the query plan and return the result set."""
        pass
