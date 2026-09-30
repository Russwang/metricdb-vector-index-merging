from core.base import QueryNode, MetricQueryProcessor
from typing import List, Any

class SQLParser:
    """
    Module 2: Metric (SQL) Query Parser (Assigned to: Student 2)
    
    Objective: Extend standard SQL parsing to support metric operators.
    Task for Student 2:
    Handle syntax like:
    SELECT * FROM table WHERE vector SIMILAR TO [0.1, 0.2, ...] AND price < 100
    """
    def parse(self, sql: str) -> QueryNode:
        """
        Parse a SQL string into a logical query plan (tree of QueryNodes).
        
        Args:
            sql: The SQL query string.
            
        Returns:
            A QueryNode representing the root of the logical plan.
        """
        # TODO [Student 2]: Implement SQL parsing logic for metric operators.
        # You may use libraries like sqlparse or build a simple regex-based parser for the demo.
        print(f"Parsing SQL: {sql}")
        return None

class QueryOptimiser:
    """
    Module 2: Logical Optimiser (Assigned to: Student 2)
    
    Objective: Develop query rewriting rules specifically for metric data.
    Task for Student 2:
    Optimization might involve:
    - Pushing predicates down.
    - Choosing whether to use a metric index or a full scan.
    - Ordering joins between relational and metric data.
    """
    def optimise(self, plan: QueryNode) -> QueryNode:
        """
        Optimize the logical plan.
        
        Args:
            plan: The initial logical plan.
            
        Returns:
            An optimized QueryNode tree.
        """
        # TODO [Student 2]: Implement logical optimization rules.
        return plan
