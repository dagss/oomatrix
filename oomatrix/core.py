
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
    

_all_impl_types = set()
_conversion_db = {}
_pending_conversion_registrations = {}

# Decorator for registering conversion. Can be used either on
# functions, or on methods in MatrixImpl instances, in which
# case the first argument is optional...
def conversion(arg1, arg2=None):
    if arg2 is None:
        dest_impl_type = arg1
        # Postpone figuring out source_type and registering to
        # the MatrixImplMetaclass
        def dec(func):
            _pending_conversion_registrations[func] = (dest_impl_type,)
            return func
        return dec
    else:
        source_impl_type, dest_impl_type = arg1, arg2
        _all_impl_types.add(source_impl_type)
        _all_impl_types.add(dest_impl_type)
        def dec(func):
            if not callable(func):
                raise TypeError("Does not decorate callable")
            add_to_graph(_conversion_db, source_impl_type, dest_impl_type, func)
            return func
        return dec

_add_operation_db = {}

# Decorator
def add_operation(source_impl_types, dest_impl_type):
    if not isinstance(source_impl_types, tuple):
        raise TypeError("source_impl_types must be a tuple")
    _all_impl_types.update(source_impl_types)
    _all_impl_types.add(dest_impl_type)
    def dec(func):
        if not callable(func):
            raise TypeError("Does not decorate callable")
        add_to_graph(_add_operation_db, source_impl_types, dest_impl_type, func)
        return func
    return dec





#
# Graph
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
    
    """
    def __init__(self, add_operations, conversions):
        self.add_operations, self.conversions = add_operations, conversions
        

    def get_vertices(self, max_node_size=4, kinds=None):
        """
        Get the vertices in the addition graph, for the matrix implementation
        types of interest. Default is all registered/loaded types.
        """
        if kinds is None:
            kinds = _all_impl_types
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
                yield set(result)

    def get_edges(self, vertex):
        # First, list all conversions
        for kind in vertex:
            for to_kind, conv_func in self.conversions.get(kind, {}).iteritems():
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
        for kind_a in vertex:
            for kind_b in vertex:
                if kind_a is kind_b:
                    continue
                add_ops = self.add_operations.get((kind_a, kind_b), {})
                for to_kind, add_func in add_ops.iteritems():
                    cost = 1
                    if to_kind is not kind_a and to_kind is not kind_b and to_kind in vertex:
                        # Result overlaps with another kind, so take into account cost
                        # of subsequent addition
                        second_add_func = self.add_operations[(to_kind, to_kind)][to_kind]
                        cost += 1
                    else:
                        second_add_func = None
                    new_vertex = vertex.difference([kind_a, kind_b]).union([to_kind])
                    yield (new_vertex, cost,
                           ('add', kind_a, kind_b, to_kind, add_func, second_add_func))

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
            target_kinds = _all_impl_types
        stop_vertices = [frozenset((v,)) for v in target_kinds]
        path = find_shortest_path(self.get_edges, start_vertex, stop_vertices)
        print path
        # Execute operations found
        matrices = reduced_operand_dict
        for action, payload in path:
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

        M, = matrices
        return M

addition_conversion_graph = AdditionGraph(_add_operation_db, _conversion_db)


#
# Core classes
#


class MatrixImplType(type):
    def __init__(cls, name, bases, dct):
        super(MatrixImplType, cls).__init__(name, bases, dct)
        # Register pending conversion registrations
        to_delete = []
        for func, decorator_args in _pending_conversion_registrations.iteritems():
            if dct.get(func.__name__, None) is func:
                dest_impl_type, = decorator_args
                conversion(cls, dest_impl_type)(func)
                to_delete.append(func)
        for func in to_delete:
            del _pending_conversion_registrations[func]

    def __repr__(cls):
        return "<kind:%s>" % cls.name

class MatrixImpl(object):
    __metaclass__ = MatrixImplType
    
    left_shape = None
    right_shape = None
    dtype = None

    def get_type(self):
        return type(self)

