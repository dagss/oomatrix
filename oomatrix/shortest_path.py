import heapq

def find_shortest_path(get_edges, start, stops):
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

    start : immutable object
        The start vertex. Can be any object usable as
        a lookup key.

    stops : set
        The possible end vertices. Algorithm stops when shortest path to one 
        of these has been found.

    Returns
    -------
    shortest_path : list
        The shortest path as a list of vertex objects. The first
        and last item are `start` and `stop`, respectively, where `stop` is
        the first endpoint reached among the possible endpoints.
    
    """
    if not callable(get_edges):
        _edges = get_edges
        def get_edges(v):
            return _edges[v]

    def resolve_path(start):
        path = []
        path.append(start)
        prev = previous[start]
        while prev is not None:
            path.append(prev)
            prev = previous[prev]
        path.reverse()
        return path

    if start in stops:
        return [start]

    curr_vertex = start
    visited = set([curr_vertex])
    considered = {curr_vertex : 0}
    previous = {curr_vertex : None}
    costheap = []
    roof = None
    while True:
        for n_vertex, cost in get_edges(curr_vertex):
            if n_vertex in visited:
                continue
            if (n_vertex not in considered or 
                considered[n_vertex] > considered[curr_vertex] + cost):
                if n_vertex in considered:
                    costheap.remove((considered[n_vertex], n_vertex))
                considered[n_vertex] = considered[curr_vertex] + cost
                previous[n_vertex] = curr_vertex
                prior_ind += 1
                heapq.heappush(costheap, (considered[n_vertex], n_vertex))
        currcost, curr_vertex = heapq.heappop(costheap)

#        #Consistency check
        if currcost != considered[curr_vertex]:
            raise ValueError("Something went wrong")
        visited.add(curr_vertex)
        if curr_vertex in stops:
            if roof is None:
                roof = currcost
                path = resolve_path(curr_vertex)
                stops.remove(curr_vertex)
            else:
                if roof == currcost:
                    raise ValueError("Ambiguous endpoint choice")
                if roof < currcost:
                    return path
        if len(costheap) == 0:
            if roof is None:
                raise ValueError("""No more nodes to visit and endpoint 
                                    not found""")
            else:
                return path
