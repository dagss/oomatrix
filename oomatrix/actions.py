"""
The tree of individual steps one must does in order to compute a
linear algebra expressions.

Each type of operation (say, multiplication of C-order with
Fortran-order dense matrices) is represented by a class which inherits
from OperationNode.

Instantiating such a class creates a node in a tree representing one
specific computation (a handle to a possible computation). The child
nodes should be arguments to the constructor, and will be new
OperationNodes.

Such trees are built by compilers (see compiler.py).  The reason for
having compilers form computation trees, rather than computing right
away, is primarily to implement `explain()` and related
features. Also, juggling such trees is useful to compilers during cost
estimation etc. This is also why we keep the tree structure, rather
than flattening it.

This is different from symbolic.py in the sense that the tree
structure *should* correspond exactly to the order computations are
done in. For instance, the operation tree would not store A * B * C,
but only (A * B) * C or A * (B * C) (unless there happens to be a
low-level optimized C/Fortran implementation of (A * B * C), like one
could envision for diagonal matrices). Also, each node in a
symbolic tree is rather shallow, while each node in the computation
tree fully describes a concrete implementation to use.
"""

class Action(object):
    """
    children - The child Action instances in the computation tree.
    """

    def __init__(self, children):
        self.children = children

    def perform(self):
        # Return MatrixImpl instance
        raise NotImplementedError()

class LeafAction(Action):
    """
    The 'leaf action' in the tree is simply to return an
    already-computed matrix
    """
    def __init__(self, matrix_impl):
        self.matrix_impl = matrix_impl

    def perform(self):
        return self.matrix_impl

    def get_kind(self):
        return type(self.matrix_impl)

class ConjugateTransposeAction(Action):
    """
    """
    
    def __init__(self, child):
        self.child = child

    def perform(self):
        child_impl = self.child.perform()
        return child_impl.conjugate_transpose()

    def get_kind(self):
        return self.child.get_kind().H

def conjugate_transpose_action(action):
    if isinstance(action, ConjugateTransposeAction):
        return action.child
    else:
        return ConjugateTransposeAction(action)

class Multiplication(Action):
    pass

class Addition(Action):
    pass

class Conversion(Action):
    pass


def conversion_action_from_function(func, source_kind, target_kind,
                                    name=None, authors=None):
    if name is None:
        name = func.__name__
    class ResultAction(Conversion):
        def __init__(self, child):
            self.child = child
            
        def perform(self):
            source = self.child.perform()
            assert isinstance(source, source_kind)
            result = func(source)
            assert isinstance(result, target_kind)
            return result

        def get_name(self):
            return name

        def get_authors(self):
            return authors

        def get_kind(self):
            return target_kind
        
    ResultAction.__name__ = 'Conversion_%s' % func.__name__
    ResultAction.__module__ = func.__module__
    return ResultAction

def multiplication_action_from_function(func, source_kinds, target_kind,
                                       name=None, authors=None):
    """
    Use to dynamically generate a Multiplication subclass
    from a function definition.

    Used in particular by the @multiplication decorator
    in core.py.
    
    """
    if name is None:
        name = func.__name__
    class ResultAction(Multiplication):
        def perform(self):
            operands = [child.perform() for child in self.children]
            assert len(operands) == len(source_kinds)
            for op, op_kind in zip(operands, source_kinds):
                assert isinstance(op, op_kind)
            result = func(*operands)
            assert isinstance(result, target_kind), '%s.%s did not return object of type %s.%s' % (
                func.__module__, func.__name__,
                target_kind.__module__, target_kind.__name__)
            return result

        def get_name(self):
            return name

        def get_authors(self):
            return authors

        def get_kind(self):
            return target_kind

    ResultAction.__name__ = 'Multiplication_%s' % func.__name__
    ResultAction.__module__ = func.__module__
    return ResultAction
                                     
        
