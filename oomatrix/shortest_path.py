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
        cost, payload)`. If not a callable, the []-operator will be used
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
        The shortest path as a list of payload objects - that is, 
        the objects representing the paths between the nodes. So if the
        shortest path has n nodes, the result will be a list of length n-1.
        If start is in the set of stops, will return None. The first item will
        be the payload from the start node to the next.
    
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

    def resolve_payload_path(start):
        path = []
        path.append(prevpay[start])
        prev = previous[start]
        while prevpay[prev] is not None:
            path.append(prevpay[prev])
            prev = previous[prev]
        path.reverse()
        return path

    stops = set(stops)

    if start in stops:
        return [None]

    visited = set([start])
    considered = {start : 0}
    previous = {start : None}
    prevpay = {start : None}
    costheap = []
    heapq.heappush(costheap, (0, start))
    roof = None
    while True:
        currcost, curr_vertex = heapq.heappop(costheap)
        visited.add(curr_vertex)

#        #Consistency check
        if currcost != considered[curr_vertex]:
            raise ValueError("Something went wrong")

        if curr_vertex in stops:
            if roof is None:
                roof = currcost
                path = resolve_payload_path(curr_vertex)
                stops.remove(curr_vertex)
            else:
                if roof == currcost:
                    raise ValueError("Ambiguous endpoint choice")
                if roof < currcost:
                    return path

        for n_vertex, cost, payload in get_edges(curr_vertex):
            if n_vertex in visited:
                continue
            if (n_vertex not in considered or 
                considered[n_vertex] > considered[curr_vertex] + cost):
                if n_vertex in considered:
                    costheap.remove((considered[n_vertex], n_vertex))
                considered[n_vertex] = considered[curr_vertex] + cost
                previous[n_vertex] = curr_vertex
                prevpay[n_vertex] = payload
                heapq.heappush(costheap, (considered[n_vertex], n_vertex))

        if len(costheap) == 0:
            if roof is None:
                raise ValueError("No more nodes to visit and endpoint "
                                 "not found")
            else:
                return path
