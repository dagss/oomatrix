import types

from .cost_value import CostValue, FLOP, MEM, MEMOP, UGLY, INVOCATION, zero_cost
from . import utils

class ImpossibleOperationError(NotImplementedError):
    pass

def return_zero_cost(*args):
    return zero_cost

class FindFlattenedPermutationTransform(object):
    
    def transform(self, root):
        self.leaves_encountered = 0
        indices = root.accept_visitor(self, root)
        return indices

    def process_children(self, node):
        child_indices = [child.accept_visitor(self, child)
                         for child in node.children]
        return child_indices
    
    def visit_add(self, node):
        permutation = utils.argsort(node.children)
        child_indices = self.process_children(node)
        sorted_child_indices = [child_indices[i] for i in permutation]
        return sum(sorted_child_indices, [])

    def visit_multiply(self, node):
        child_indices = self.process_children(node)
        return sum(child_indices, [])

    def visit_single_child(self, node):
        return node.child.accept_visitor(self, node.child)

    visit_conjugate_transpose = visit_inverse = visit_single_child
    visit_factor = visit_decomposition = visit_single_child

    def visit_kind(self, node):
        idx = self.leaves_encountered
        self.leaves_encountered += 1
        return [idx]
        

class Computation(object):
    """
    Wraps a computation function to provide metadata, and enable calling it
    in a standardized fashion.

    Compares as `object`
    """
    
    def __init__(self, callable, match_expression, target_kind,
                 name=None, cost_callable=None):
        permutation = FindFlattenedPermutationTransform().transform(match_expression)
        self.call_permutation = utils.invert_permutation(permutation)        
        self.callable = callable
        self.match_expression = match_expression
        self.target_kind = target_kind
        self.name = (name if name is not None
                     else '%s.%s' % (callable.__module__, callable.__name__))
        self.cost_callable = cost_callable

    def compute(self, matrices):
        reordered_matrices = [matrices[i] for i in self.call_permutation]
        return self.callable(*reordered_matrices)

    def get_cost(self, meta_args):
        if self.cost_callable is None:
            raise AssertionError('The cost of %s is not assigned' % self.name)
        cost = self.cost_callable(*meta_args)
        if cost == 0:
            cost = zero_cost
        if not isinstance(cost, CostValue):
            raise TypeError('cost function %s for %s did not return 0 or a '
                            'CostValue' % (self.cost_callable, self.callable))
        return cost + INVOCATION

    def __call__(self, *args, **kw):
        """A direct call of the computation function

        Since this is meant for 'manual' use, there's no reordering/sorting
        of arguments
        """
        return self.callable(*args, **kw)

    def __repr__(self):
        return '<Computation:%s>' % self.name

   

def register_computation(match, target_kind, obj):
    match.universe.join_with(target_kind.universe)
    match.universe.add_computation(match, target_kind, obj)

def register_conversion(from_kind, to_kind, obj):
    register_computation(from_kind, to_kind, obj)

def computation(match, target_kind, name=None, cost=None):
    if cost is None:
        raise ValueError('Must provide a cost')
    if isinstance(cost, CostValue) or cost == 0:
        _cost = lambda *args: cost
    else:
        _cost = cost
    def dec(obj):
        computation = Computation(obj, match, target_kind, name, _cost)
        register_computation(match, target_kind, computation)
        return computation
    return dec


def conversion_method(target_kind, name=None, cost=None):
    from .kind import add_post_class_definition_hook
    def dec(method):
        def doit(cls, method):
            return computation(cls, target_kind, name=name, cost=cost)(method)
        add_post_class_definition_hook(doit, method)
        return method
    return dec

def conversion(arg1, arg2=None, name=None, cost=None):
    """Decorator for registering conversions
    
    A @conversion is really a computation from one kind to another::

        @computation(Diagonal, Dense)
        def diagonal_to_dense(x): ...

    However, we use a different decorator for readability purposes. Also,
    @conversion can be used on methods, and will in that case use the
    class as the first argument
    """
    if arg2 is None:
        return conversion_method(arg1, name=name, cost=cost)
    else:
        return computation(arg1, arg2, name=name, cost=cost)
        

        
