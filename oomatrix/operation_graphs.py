"""
Provides lookup of addition and multiplication operations combined with
conversion operationsl. Dijkstra's shortest-path is used for this.

One may well be able to get rid of this in the future and replace it with a
more generic approach...
"""


from .graph.shortest_path import find_shortest_path
from . import kind
from .kind import AddPatternNode, MultiplyPatternNode
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
        # vertex is a set of kinds
        universe = iter(vertex).next().universe
        # First, list all conversions
        for kind in vertex:
            conversions = universe.get_computations(kind.get_key())
            for to_kind, conversion_list in conversions.iteritems():
                conversion = conversion_list[0] # just take the first for now
                cost = 1
                if to_kind in vertex:
                    # Converting to another kind that is already in vertex,
                    # so take into account addition immediately
                    second_add = universe.get_computations(
                        AddPatternNode([to_kind, to_kind]).get_key())[to_kind][0]
                    cost += 1
                else:
                    second_add = None
                new_vertex = vertex.difference([kind]).union([to_kind])
                yield (new_vertex, cost, ('convert', kind, to_kind, conversion,
                                          second_add))
                    
        # Then, list any cross-kind additions
        a_visited = set()
        for kind_a in vertex:
            a_visited.add(kind_a)
            for kind_b in vertex:
                if kind_b in a_visited:
                    continue
                add_ops = universe.get_computations(
                    AddPatternNode([kind_a, kind_b]).get_key())
                for to_kind, add_list in add_ops.iteritems():
                    cost = 1
                    if to_kind is not kind_a and to_kind is not kind_b and to_kind in vertex:
                        # Result overlaps with another kind, so take into account cost
                        # of subsequent addition
                        second_add = universe.get_computations(
                            AddPatternNode([to_kind, to_kind]).get_key())[to_kind][0]
                        cost += 1
                    else:
                        second_add = None
                    new_vertex = vertex.difference([kind_ap, kind_bp]).union([to_kind])
                    yield (new_vertex, cost,
                           ('add', kind_ap, kind_bp, to_kind, add_func, second_add))

    def find_cheapest_action(self, operands, target_kinds=None):
        operand_dict = {}
        # Sort operands by their kind
        for op in operands:
            operand_dict.setdefault(op.kind, []).append(op)

        universe = iter(operand_dict.keys()).next().universe

        # Perform all within-kind additions
        reduced_operand_dict = {}
        for kind, operand_list in operand_dict.iteritems():
            try:
                additions = universe.get_computations(
                    AddPatternNode([kind, kind]).get_key())
                # just take the first one
                addition = additions[kind][0]
            except (KeyError, IndexError):
                raise AssertionError(
                    'Within-kind matrix addition not '
                    'defined for %s, this is a bug' % kind)
            
            u, rest = operand_list[0], operand_list[1:]
            for v in rest:
                u = Computable(addition,
                               children=[u, v],
                               nrows=u.nrows,
                               ncols=v.ncols,
                               dtype=u.dtype)
            reduced_operand_dict[kind] = u
        del operand_dict

        if len(reduced_operand_dict) == 1:
            # Homogenous addition, do early return
            M, = reduced_operand_dict.values()
            return M

        # Find shortest path through graph to target_kinds
        start_vertex = frozenset(reduced_operand_dict) # get set of keys
        if target_kinds is None:
            target_kinds = universe.get_kinds()
        stop_vertices = [frozenset((v,)) for v in target_kinds]
        path = find_shortest_path(self.get_edges, start_vertex, stop_vertices)
        # Execute operations found
        operands = reduced_operand_dict
        for payload in path:
            edge, args = payload[0], payload[1:]
            if edge == 'add':
                kind_a, kind_b, to_kind, addition, second_addition = args
                A = operands.pop(kind_a)
                B = operands.pop(kind_b)
                computable = Computable(addition,
                                        children=[A, B],
                                        nrows=A.nrows,
                                        ncols=A.ncols,
                                        dtype=A.dtype # todo
                                        )
            elif edge == 'convert':
                kind, to_kind, conversion, second_addition = args
                child = operands.pop(kind)
                computable = Computable(conversion,
                                        children=[child],
                                        nrows=child.nrows,
                                        ncols=child.ncols,
                                        dtype=child.dtype # todo
                                        )
            assert (to_kind in operands) == (second_addition is not None)
            if second_addition is not None:
                computable = Computable(second_addition,
                                        children=[computable,
                                                  operands[to_kind]],
                                        nrows=computable.nrows,
                                        ncols=computable.ncols,
                                        dtype=computable.dtype # todo
                                        )
            operands[to_kind] = computable

        computable, = operands.values()
        return computable

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
            if target_kinds != universe.get_kinds():
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

