
from .graph.shortest_path import find_shortest_path
from . import actions

class ImpossibleOperationError(NotImplementedError):
    pass

# TODO Cost model etc.

# Graphs are represented as
#   { source_key : { dest_key : value } }
# TODO When costs are implemented, value -> [value, ...]
#
# source_key is a single type for conversions, tuple of
# multiple types for operations

def add_to_graph(d, source_key, target_key, value):
    try:
        x = d[source_key]
    except KeyError:
        x = {}
        d[source_key] = x
    if target_key in x:
        # We should really allow more than one method and select based
        # on cost
        raise NotImplementedError("TODO, requires proper cost calculation")
    x[target_key] = value
    


#
# Operation graphs
#

class ConversionGraph(object):
    # Functions in self.listeners will be called whenever
    # a conversion is added.
    
    # global shared class variable:
    _global_pending_conversion_registrations = {}

    def __init__(self):
        self.conversions = {}
        self.all_kinds = set()
        self.listeners = []

    #
    # Decorators
    #
    def conversion_method(self, dest_kind):
        # Postpone figuring out source_type and registering to
        # the MatrixImplMetaclass
        def dec(func):
            ConversionGraph._global_pending_conversion_registrations[func] = (
                self, dest_kind)
            return func
        return dec
        
    def conversion_decorator(self, arg1, arg2=None):
        if arg2 is None:
            return self.conversion_method(arg1)
        source_kind, dest_kind = arg1, arg2        
        self.all_kinds.add(source_kind)
        self.all_kinds.add(dest_kind)
        def dec(func):
            if not callable(func):
                raise TypeError("Does not decorate callable")
            action_type = actions.conversion_action_from_function(func,
                                                                  source_kind,
                                                                  dest_kind)
            add_to_graph(self.conversions, source_kind, dest_kind, action_type)
            for listener in self.listeners:
                listener(self, source_kind, dest_kind, func)
            return func
        return dec

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

    def __init__(self, conversion_graph):
        self.conversion_graph = conversion_graph
        self.add_operations = {}

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
            operand_dict.setdefault(op.get_kind(), []).append(op)

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

    # Decorator
    def addition_decorator(self, source_kinds, dest_kind):
        if not isinstance(source_kinds, tuple):
            raise TypeError("source_kinds must be a tuple")
        self.conversion_graph.all_kinds.update(source_kinds)
        self.conversion_graph.all_kinds.add(dest_kind)
        def dec(func):
            if not callable(func):
                raise TypeError("Does not decorate callable")
            A, B = source_kinds

            if B < A:
                # Reverse the arguments
                def reverse_arguments(a, b):
                    return func(b, a)
                func = reverse_arguments
                A, B = B, A
            
            if (A, B) in self.add_operations:
                raise Exception("Already registered addition for %s" % (A, B))

            action_factory = actions.addition_action_from_function(func,
                                                                   source_kinds,
                                                                   dest_kind)
            add_to_graph(self.add_operations, source_kinds, dest_kind, action_factory)
            return func
        return dec

class MatVecGraph(object):
    def __init__(self, conversion_graph):
        self.conversion_graph = conversion_graph
        conversion_graph.listeners.append(self.on_conversion_added)

    def on_conversion_added(self, conversion_graph, source_kind, dest_kind, func):
        if hasattr(dest_kind, 'apply'):
            pass

    # Decorator
    def matvec(self, matrix_impl, vec):
        raise NotImplementedError()


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

    def __init__(self, conversion_graph):
        self.conversion_graph = conversion_graph
        self.multiply_operations = {}

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
        conversions = self.conversion_graph.conversions

        if len(vertex) > 1:
            # Multiplication actions at this point
            for target_kind, mul_action_factory in self.multiply_operations.get(
                vertex, {}).iteritems():
                yield ((target_kind,), 1, ('multiply', mul_action_factory))

        # List possible conversions of one of the elements of the tuple
        # This can happen both before multiplication (len > 1) and after (len == 1)
        for i in range(len(vertex)):
            from_kind = vertex[i]
            for target_kind, conversion_factory in conversions.get(from_kind, {}).iteritems():
                neighbour = tuple_replace(vertex, i, target_kind)
                yield (neighbour, 1, (i, conversion_factory))

    # Public-facing methods
    def find_cheapest_action(self, children, target_kinds=None):
        assert all(isinstance(child, actions.Action)
                   for child in children)
        assert len(children) == 2
        
        # First, operate on the kinds of the child actions
        start_vertex = tuple(x.get_kind() for x in children)
        if target_kinds is None:
            target_kinds = self.conversion_graph.all_kinds
        stop_vertices = [(v,) for v in target_kinds]
        try:
            path = find_shortest_path(self.get_edges,
                                      start_vertex, stop_vertices)
        except ValueError:
            if target_kinds != self.conversion_graph.all_kinds:
                postfix = ' to produce one of [%s]' % (
                    ', '.join(str(kind) for kind in target_kinds))
            else:
                postfix = ''
            raise ImpossibleOperationError(
                "Found no way of multiplying %r with %r%s" %
                (start_vertex + (postfix,)))

        node = tuple(children)
        for edge, action_factory in path:
            if edge == 'multiply':
                node = (action_factory(node),)
            else:
                # edge is an int indicating which element to convert with
                # the action
                node = node[:edge] + (action_factory(node[edge]),) + node[edge + 1:]
        assert len(node) == 1
        return node[0]


    # Decorator
    def multiplication_decorator(self, source_kinds, dest_kind):
        if not isinstance(source_kinds, tuple) or len(source_kinds) != 2:
            raise TypeError("source_kinds must be a tuple of length 2")
        self.conversion_graph.all_kinds.update(source_kinds)
        self.conversion_graph.all_kinds.add(dest_kind)
        def dec(func):
            if not callable(func):
                raise TypeError("Does not decorate callable")
            if source_kinds in self.multiply_operations:
                raise Exception("Already registered multiplication for %s" % (source_kinds,))
            action_factory = actions.multiplication_action_from_function(func,
                                                                         source_kinds,
                                                                         dest_kind)
            add_to_graph(self.multiply_operations, source_kinds, dest_kind, action_factory)
            return func
        return dec


# Create default operation graph, and define some decorators
# as methods bounds on this graph instance
conversion_graph = ConversionGraph()
conversion = conversion_graph.conversion_decorator

addition_conversion_graph = AdditionGraph(conversion_graph)
addition = addition_conversion_graph.addition_decorator


multiply_graph = MultiplyPairGraph(conversion_graph)
multiplication = multiply_graph.multiplication_decorator


