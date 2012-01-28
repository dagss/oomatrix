
from .graph.shortest_path import find_shortest_path
from . import kind
from .computation import Computable, BaseComputable

class ImpossibleOperationError(NotImplementedError):
    pass


#
# Operation graphs
#

class AdditionGraph(object):
    """
    Implements the graph resulting from combining registered
    addition operations and conversion operations, so that the most
    efficient way of adding matrices can be found.

    Since addition is commutative this graph is exponentially big in
    the number of terms to add. Therefore we use the following
    heuristic:

      - Every kind of matrix has a direct implementation of addition
        with the same kind
      - Adding together all matrices of the same kind first is optimal

    The graph is now exponential-sized in the number of registered
    matrix kinds. However it is very sparse, as the number of edges in
    total is bound by the number of registered conversions/additions
    times the number of matrix kinds.

    So we first reduce the input operands to one input operand per kind,
    and then use the shortest path through the graph to work on the
    remaining operands.

    The nodes in the graph are frozensets of matrix kinds. When the
    number of items in the frozenset reaches 1, all additions have
    been performed. Conversions move between different sets of same
    size, addition operations move to sets having one less item.

    The arguments for add operations are wrapped so that they can be
    called with arguments sorted in the order of their kind.
    
    """

    def get_vertices(self, max_node_size=4, kinds=None):
        """
        Get the vertices in the addition graph, for the matrix implementation
        types of interest. Default is all registered/loaded types.
        """
        if kinds is None:
            kinds = self.conversion_graph.all_kinds
        kinds = list(kinds)

        def all_subsets(size, itemlst):
            if size == 0 or len(itemlst) == 0:
                yield ()
            else:
                for i in range(len(itemlst) - size + 1):
                    u, rest = itemlst[i], itemlst[i + 1:]
                    for subset in all_subsets(size - 1, rest):
                        yield (u,) + subset

        n = min(max_node_size, len(kinds))
        for size in range(1, n + 1):
            for result in all_subsets(size, kinds):
                yield frozenset(result)

    def get_edges(self, vertex):
        # First, list all conversions
        conversions = self.conversion_graph.conversions
        for kind in vertex:
            for to_kind, conv_func in conversions.get(kind, {}).iteritems():
                cost = 1
                if to_kind in vertex:
                    # Converting to another kind that is already in vertex,
                    # so take into account addition immediately
                    second_add_func = self.add_operations[(to_kind, to_kind)][to_kind]
                    cost += 1
                else:
                    second_add_func = None
                new_vertex = vertex.difference([kind]).union([to_kind])
                yield (new_vertex, cost, ('convert', kind, to_kind, conv_func,
                                          second_add_func))
                    
        # Then, list any cross-kind additions
        a_visited = set()
        for kind_a in vertex:
            a_visited.add(kind_a)
            for kind_b in vertex:
                if kind_b in a_visited:
                    continue
                if kind_b < kind_a:
                    kind_ap, kind_bp = kind_b, kind_a
                else:
                    kind_ap, kind_bp = kind_a, kind_b
                add_ops = self.add_operations.get((kind_ap, kind_bp), {})
                for to_kind, add_func in add_ops.iteritems():
                    cost = 1
                    if to_kind is not kind_ap and to_kind is not kind_bp and to_kind in vertex:
                        # Result overlaps with another kind, so take into account cost
                        # of subsequent addition
                        second_add_func = self.add_operations[(to_kind, to_kind)][to_kind]
                        cost += 1
                    else:
                        second_add_func = None
                    new_vertex = vertex.difference([kind_ap, kind_bp]).union([to_kind])
                    yield (new_vertex, cost,
                           ('add', kind_ap, kind_bp, to_kind, add_func, second_add_func))

    def find_cheapest_action(self, operands, target_kinds=None):
        operand_dict = {}
        # Sort operands by their kind
        for op in operands:
            operand_dict.setdefault(type(op), []).append(op)

        # Perform all within-kind additions
        reduced_operand_dict = {}
        for kind, operand_list in operand_dict.iteritems():
            try:
                add_action_factory = self.add_operations[
                    (kind, kind)][kind]
            except KeyError:
                raise AssertionError(
                    'Within-kind matrix addition not '
                    'defined for %s, this is a bug' % kind)
            
            u, rest = operand_list[0], operand_list[1:]
            for v in rest:
                u = add_action_factory([u, v])
            reduced_operand_dict[kind] = u
        del operand_dict

        if len(reduced_operand_dict) == 1:
            # Homogenous addition, do early return
            M, = reduced_operand_dict.values()
            return M

        # Find shortest path through graph to target_kinds
        start_vertex = frozenset(reduced_operand_dict) # get set of keys
        if target_kinds is None:
            target_kinds = self.conversion_graph.all_kinds
        stop_vertices = [frozenset((v,)) for v in target_kinds]
        path = find_shortest_path(self.get_edges, start_vertex, stop_vertices)
        # Execute operations found
        matrices = reduced_operand_dict
        for payload in path:
            edge, args = payload[0], payload[1:]
            if edge == 'add':
                kind_a, kind_b, to_kind, add_action_factory, second_add_action_factory = args
                A = matrices.pop(kind_a)
                B = matrices.pop(kind_b)
                C = add_action_factory([A, B])
            elif edge == 'convert':
                kind, to_kind, conv_action_factory, second_add_action_factory = args
                A = matrices.pop(kind)
                C = conv_action_factory(A)
            assert (to_kind in matrices) == (second_add_action_factory is not None)
            if second_add_action_factory is not None:
                C = second_add_action_factory([C, matrices[to_kind]])
            matrices[to_kind] = C

        M, = matrices.values()
        return M

def tuple_replace(tup, idx, value):
    return tup[:idx] + (value,) + tup[idx + 1:]

class MultiplyPairGraph(object):
    """
    A graph used to find the best combination of conversions to
    invoke a single matrix multiplication ``A * B``. If A
    and B are of the same type, this is often just the multiplication
    operation itself, but things can get more complicated if one
    or several conversions must be performed.

    This graph does not address the best way to form the product of
    more than two matrices. Doing that is a classical example of dynamic
    programming which can use this class in its inner step.
    """

    # Implement graph on which to perform shortest-path
    def get_vertices(self, max_node_size=4, kinds=None):
        """
        Get the vertices in the graph, for the matrix implementation
        types of interest. Default is all registered/loaded types.
        """
        if kinds is None:
            kinds = self.conversion_graph.all_kinds
        kinds = list(kinds)

        for A_kind in kinds:
            for B_kind in kinds:
                yield (A_kind, B_kind) # source vertex
            yield (A_kind,) # target vertex

    def get_edges(self, vertex):
        # Payload: ('multiply'|0|1, func)
        # Where 0, 1 denotes conversion of left or right operand
        universe = vertex[0].universe

        if len(vertex) > 1:
            # Multiplication actions at this point
            muls = universe.get_computations(
                kind.MultiplyPatternNode(vertex).get_key())
            for target_kind, computation_list in muls.iteritems():
                # just take the first one for now
                yield ((target_kind,), 1, ('multiply', computation_list[0]))

        # List possible conversions of one of the elements of the tuple
        # This can happen both before multiplication (len > 1) and after (len == 1)
        for i in range(len(vertex)):
            from_kind = vertex[i]
            convs = universe.get_computations(from_kind.get_key())
            for target_kind, conversion_list in convs.iteritems():
                neighbour = tuple_replace(vertex, i, target_kind)
                # just take the first one for now
                yield (neighbour, 1, (i, conversion_list[0]))

    # Public-facing methods
    def find_cheapest_action(self, children, target_kinds=None):
        assert all(isinstance(child, BaseComputable)
                   for child in children)
        assert len(children) == 2
        universe = children[0].kind.universe
        
        # First, operate on the kinds of the child actions
        start_vertex = tuple(x.kind for x in children)
        if target_kinds is None:
            target_kinds = universe.get_kinds()
        stop_vertices = [(v,) for v in target_kinds]
        try:
            path = find_shortest_path(self.get_edges,
                                      start_vertex, stop_vertices)
        except ValueError:
            # TODO!
            if target_kinds != self.conversion_graph.all_kinds:
                postfix = ' to produce one of [%s]' % (
                    ', '.join(str(kind) for kind in target_kinds))
            else:
                postfix = ''
            raise ImpossibleOperationError(
                "Found no way of multiplying %r with %r%s" %
                (start_vertex + (postfix,)))

        vertex = tuple(children)
        for action, computation in path:
            if action == 'multiply':
                computable = Computable(computation,
                                        children=vertex,
                                        nrows=vertex[0].nrows,
                                        ncols=vertex[-1].ncols,
                                        dtype=vertex[0].dtype # todo
                                        )
                vertex = (computable,)
            else:
                # action is an int indicating which element to convert with
                # the computation
                idx = action
                node = vertex[idx]
                convertable = Computable(computation,
                                         children=(node,),
                                         nrows=node.nrows,
                                         ncols=node.ncols,
                                         dtype=node.dtype # todo
                                         )
                vertex = vertex[:idx] + (convertable,) + vertex[idx + 1:]
        assert len(vertex) == 1
        return vertex[0]


addition_graph = AdditionGraph()
multiplication_graph = MultiplyPairGraph()

