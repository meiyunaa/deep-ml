import numpy as np
from heapq import nsmallest
def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    def distance_squared(point):
        return sum((a - b) ** 2 for a, b in zip(point, query_point))
    neighbors = nsmallest(k, points, key=distance_squared)
    return neighbors
    

