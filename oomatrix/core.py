
from .shortest_path import find_shortest_path

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
    def conversion_method(self, dest_impl_type):
        # Postpone figuring out source_type and registering to
        # the MatrixImplMetaclass
        def dec(func):
            ConversionGraph._global_pending_conversion_registrations[func] = (
                self, dest_impl_type)
            return func
        return dec
        
    def conversion(self, arg1, arg2=None):
        if arg2 is None:
            return self.conversion_method(arg1)
        source_impl_type, dest_impl_type = arg1, arg2        
        self.all_kinds.add(source_impl_type)
        self.all_kinds.add(dest_impl_type)
        def dec(func):
            if not callable(func):
                raise TypeError("Does not decorate callable")
            add_to_graph(self.conversions, source_impl_type, dest_impl_type, func)
            for listener in self.listeners:
                listener(self, source_impl_type, dest_impl_type, func)
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

    def perform(self, operands, target_kinds=None):
        operand_dict = {}
        # Sort operands by their kind
        for op in operands:
            operand_dict.setdefault(op.get_type(), []).append(op)

        # Perform all within-kind additions
        reduced_operand_dict = {}
        for kind, operand_list in operand_dict.iteritems():
            add_operation = self.add_operations[(kind, kind)][kind]
            u, rest = operand_list[0], operand_list[1:]
            for v in rest:
                u = add_operation(u, v)
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
            action, args = payload[0], payload[1:]
            if action == 'add':
                kind_a, kind_b, to_kind, add_func, second_add_func = args
                A = matrices.pop(kind_a)
                B = matrices.pop(kind_b)
                C = add_func(A, B)
            elif action == 'convert':
                kind, to_kind, conv_func, second_add_func = args
                A = matrices.pop(kind)
                C = conv_func(A)
            assert (to_kind in matrices) == (second_add_func is not None)
            if second_add_func is not None:
                C = second_add_func(C, matrices[to_kind])
            matrices[to_kind] = C

        M, = matrices.values()
        return M

    # Decorator
    def add_operation(self, source_impl_types, dest_impl_type):
        if not isinstance(source_impl_types, tuple):
            raise TypeError("source_impl_types must be a tuple")
        self.conversion_graph.all_kinds.update(source_impl_types)
        self.conversion_graph.all_kinds.add(dest_impl_type)
        def dec(func):
            if not callable(func):
                raise TypeError("Does not decorate callable")
            A, B = source_impl_types

            if B < A:
                # Reverse the arguments
                print "Reversing", A, B
                def reverse_arguments(a, b):
                    return func(b, a)
                func = reverse_arguments
                A, B = B, A
            
            if (A, B) in self.add_operations:
                raise Exception("Already registered addition for %s" % (A, B))
            add_to_graph(self.add_operations, source_impl_types, dest_impl_type, func)
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


# Create default operation graph, and define some decorators
# as methods bounds on this graph instance
conversion_graph = ConversionGraph()
conversion = conversion_graph.conversion

addition_conversion_graph = AdditionGraph(conversion_graph)
add_operation = addition_conversion_graph.add_operation


#
# Core classes
#


class MatrixImplType(type):
    def __init__(cls, name, bases, dct):
        super(MatrixImplType, cls).__init__(name, bases, dct)
        # Register pending conversion registrations
        to_delete = []
        pending = ConversionGraph._global_pending_conversion_registrations
        for func, decorator_args in pending.iteritems():
            if dct.get(func.__name__, None) is func:
                graph, dest_impl_type = decorator_args
                graph.conversion(cls, dest_impl_type)(func)
                to_delete.append(func)
        for func in to_delete:
            del pending[func]

    def __repr__(cls):
        return "<kind:%s>" % cls.name

    def __eq__(cls, other_cls):
        return cls is other_cls

    def __ne__(cls, other_cls):
        return cls is not other_cls

    def __cmp__(cls, other_cls):
        if not isinstance(other_cls, MatrixImplType):
            raise TypeError("Invalid comparison")
        return cmp(cls.name, other_cls.name)

class MatrixImpl(object):
    __metaclass__ = MatrixImplType
    
    left_shape = None
    right_shape = None
    dtype = None

    def get_type(self):
        return type(self)

