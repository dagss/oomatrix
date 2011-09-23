
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
_conversion_graph = {}
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
            add_to_graph(_conversion_graph, source_impl_type, dest_impl_type, func)
            return func
        return dec

_add_operation_graph = {}

# Decorator
def add_operation(source_impl_types, dest_impl_type):
    if not isinstance(source_impl_types, tuple):
        raise TypeError("source_impl_types must be a tuple")
    _all_impl_types.update(source_impl_types)
    _all_impl_types.add(dest_impl_type)
    def dec(func):
        if not callable(func):
            raise TypeError("Does not decorate callable")
        add_to_graph(_add_operation_graph, source_impl_types, dest_impl_type, func)
        return func
    return dec





#
# Graph
#

class AdditionGraph(object):
    def __init__(self, add_graph, conversion_graph):
        self.add_graph, self.conversion_graph = add_graph, conversion_graph

    def get_vertices(self, impl_types=None):
        """
        Get the vertices in the addition graph, for the matrix implementation
        types of interest. Default is all registered/loaded types.
        """
        if impl_types is None:
            impl_types = _all_impl_types
        for A in impl_types:
            for B in impl_types:
                yield (A, B) # source operands
            yield A # computed targe operands

    def get_edges(self, vertex):
        if isinstance(vertex, tuple):
            # Has not yet applied to operation. First, provide all
            # add-operations
            A, B = vertex
            ops = self.add_graph.get((A, B), None)
            if ops is not None:
                for target_type, func in ops.iteritems():
                    yield (target_type, 1, ('operation', func))
            # Then, provide all conversions
            if A != B:
                # Addition is commutative
                yield ((B, A), 1, ('flip', None))
            # All possible conversions of A or B (A *and* B simply goes
            # through the other paths)
            for possible_A_type, func in self.conversion_graph.get(A, {}).iteritems():
                yield ((possible_A_type, B), 1, (0, func))
            for possible_B_type, func in self.conversion_graph.get(A, {}).iteritems():
                yield ((A, possible_B_type), 1, (1, func))
        else:
            # Has applies operation. Only conversions remain
            A = vertex
            for possible_A_type, func in self.conversion_graph.get(A, {}).iteritems():
                yield (possible_A_type, 1, ('post_conversion', func))

    def resolve(self, source_impl_types, target_impl_types=None):
        if target_impl_types is None:
            target_impl_types = _all_impl_types
        path = find_shortest_path(self.get_edges,
                                  source_impl_types,
                                  target_impl_types)
        return path

    def perform(self, sources, target_impl_types=None):
        sources = list(sources)
        path = self.resolve(tuple([type(source) for source in sources]),
                            target_impl_types)
        vertex = sources
        for action, func in path:
            if action == 'flip':
                vertex = vertex[::-1]
            elif action == 'post_conversion':
                vertex = func(vertex)
            elif action == 'operation':
                vertex = func(*vertex)
            else:
                # action is integer indicating item to convert
                vertex[action] = func(vertex[action])
        return vertex

                
            
        
        
            
            
        



addition_conversion_graph = AdditionGraph(_add_operation_graph, _conversion_graph)




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
        return "<type:%s>" % cls.__name__

class MatrixImpl(object):
    __metaclass__ = MatrixImplType
    
    left_shape = None
    right_shape = None
    dtype = None

    def get_type(self):
        return type(self)

