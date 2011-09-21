

def find_shortest_path(get_edges, start, stop):
    """
    Finds a shortest path between two vertices in a graph.

    Uses Dijsktra's algorithm, although without a priority queue,
    for a complexity of O(|V|^2).

    Parameters
    ----------
    get_edges : callable or dict-like
        Describes the graph.  If a callable, it should take a vertex
        as its single argument and return an iterable (possibly empty)
        listing the edges going from the node, as tuples `(vertex,
        cost)`. If not a callable, the []-operator will be used
        instead.

    start, stop : immutable object
        The start and stop vertices. Can be any object usable as
        a lookup key.

    Returns
    -------
    shortest_path : list
        The shortest path as a list of vertex objects. The first
        and last item are `start` and `stop`, respectively.
    
    """
    if not callable(get_edges):
        _edges = get_edges
        def get_edges(v):
            return _edges[v]
    
    raise NotImplementedError()
