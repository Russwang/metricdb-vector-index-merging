from core.base import MetricRelation
from typing import List, Tuple
import numpy as np

class EntityLinker:
    """
    Module 5: Geometric Hashing for Entity Linking (Assigned to: Student 5)
    
    Objective: Discover reference mappings (anchor points) between different 
    MetricRelations without relying on shared unique identifiers (e.g., across 
    different platforms or embedding spaces).
    
    Task for Student 5:
    Implement geometric hashing techniques to find entities that likely represent 
    the same real-world object based on their relative metric positions.
    """
    def find_anchor_points(self, rel_a: MetricRelation, rel_b: MetricRelation) -> List[Tuple[str, str]]:
        """
        Identify a small set of high-confidence matches (anchors) between two relations.
        
        Args:
            rel_a: The first MetricRelation.
            rel_b: The second MetricRelation.
            
        Returns:
            A list of ID pairs [(id_a, id_b), ...] that are linked.
        """
        # TODO [Student 5]: Implement geometric hashing to link vectors encoding the same objects
        return []

    def link_entities(self, rel_a: MetricRelation, rel_b: MetricRelation):
        """
        Perform full entity linking across two databases using discovered anchors.
        
        Args:
            rel_a: The first MetricRelation.
            rel_b: The second MetricRelation.
        """
        # TODO [Student 5]: Perform entity linking across databases
        pass
