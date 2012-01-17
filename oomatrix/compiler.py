import sys
import numpy as np

# TODO: Computers should be reentrant/thread-safe, since they can
# be assigned to a global configuration variable.

from . import formatter, symbolic, actions
from .matrix import Matrix
from .core import ImpossibleOperationError

def is_right_vector(expr):
    return expr.ncols == 1 and expr.nrows > 1

def is_left_vector(expr):
    return expr.nrows == 1 and expr.ncols > 1

class SimplisticCompilation(object):
    def __init__(self, multiply_graph, add_graph):
        self.multiply_graph = multiply_graph
        self.add_graph = add_graph

    def compile(self, symbolic_node, target_kinds=None):
        assert isinstance(symbolic_node, symbolic.ExpressionNode), (
            '%r is not symbolic node' % symbolic_node)
        action_node = symbolic_node.accept_visitor(
            self, symbolic_node, target_kinds=target_kinds)
        assert isinstance(action_node, actions.Action)
        return action_node

    def visit_add(self, expr, target_kinds):
        # Simply use self.add_graph; which deals with any number
        # of operands. But we do try both transposed and
        # non-transposed. (TODO: This should perhaps be refactored
        # to add_graph, since any *pair* could be pre- and
        # post-transposed)
        children = [self.compile(child) for child in expr.children]
        try:
            result = self.add_graph.find_cheapest_action(
                children, target_kinds)
        except ImpossibleOperationError:
            # Try the transpose sum
            transposed_children = [
                actions.conjugate_transpose_action(child)
                for child in children]
            result = self.add_graph.find_cheapest_action(
                transposed_children, target_kinds)
            result = acions.conjugate_transpose_action(result)
        return result

    def multiply_pair(self, left, right, target_kinds):
        # Perform multiplications
        try:
            node = self.multiply_graph.find_cheapest_action(
                (left, right), target_kinds=target_kinds)
        except ImpossibleOperationError:
            # Try the transpose product
            left = actions.conjugate_transpose_action(left)
            right = actions.conjugate_transpose_action(right)
            node = self.multiply_graph.find_cheapest_action(
                (right, left))
            node = actions.conjugate_transpose_action(node)
        return node

    def visit_multiply(self, expr, target_kinds):
        # First, use the distributive law on expr
        # (this may create a lot of common subexpressions)
        # TODO: Eliminate common subexpressions (in a *seperate*
        # path which also picks up such created by user)
        old_expr = expr
        if is_right_vector(expr):
            expr = symbolic.apply_right_distributive_rule(expr)
        elif is_left_vector(expr):
            expr = symbolic.apply_left_distributive_rule(expr)
        if not isinstance(expr, symbolic.MultiplyNode):
            # Did apply distributive rule and we now have an AddNode,
            # so we don't want to be in visit_multiply. Otherwise,
            # continue, so that we avoid infinite recursion
            return self.compile(expr, target_kinds)
            
        # OK, now we first compute each element, and then do the
        # product 2 and 2 terms, either right-to-left or left-to-right
        assert len(expr.children) >= 2
        children = [self.compile(x) for x in expr.children]
        
        # Figure out if this is a "matrix-vector" product, in which
        # case we change the order of multiplication.
        if is_right_vector(expr):
            # Right-to-left
            right = children[-1]
            for left in children[-2::-1]:
                right = self.multiply_pair(left, right, target_kinds)
            return right
        else:
            # Left-to-right
            left = children[0]
            for right in children[1:]:
                left = self.multiply_pair(left, right, target_kinds)
            return left

    def visit_leaf(self, expr, target_kinds):
        if target_kinds is not None:
            raise NotImplementedError()
        return actions.LeafAction(expr.matrix_impl)
            
    def visit_inverse(self, expr, target_kinds):
        raise NotImplementedError()

    def visit_conjugate_transpose(self, expr, target_kinds): 
        if target_kinds is not None:
            raise NotImplementedError()
        child_action = self.compile(expr.child)
        return actions.ConjugateTransposeAction(child_action)

    def visit_bracket(self, expr, target_kinds):
        return self.compile(expr.child, target_kinds=expr.kinds)


class SimplisticCompiler(object):
    """
    Compiles an expression using some simple syntax-level rules.
    First, we treat all matrices with one 1-length dimension as
    a "vector". Then, ignoring any cost estimates (A and B
    are matrices, u is a vector):

      - Matrix-vector products are performed such that there's always
        a vector; ``A * B * x`` is performed right-to-left and
        ``x * A * B`` is performed left-to-right. Similarly,
        ``(A + B) * x`` is computed as ``A * x + B * x``.
        
      - Matrix-matrix products such as ``A * B * C`` are performed
        left-to-right (no matter what). Also, expressions are computed
        as formed: ``(A + B) * X`` first computes ``A + B`` before
        multiplying with ``X``.

      - ``A.h * u`` is computed as per the above rules
        (conjugate_transpose() being cheap), however, if that
        operation is not possible, ``(u.h * A).h`` is attempted
        instead before giving up.

      - Matrix additions ``A + B`` are performed in some arbitrary
        order.

      - ``A.i * B`` always first attempts ``A.solve_right(B)``,
        then ``A.inverse() * B``.

      - ``A.h.i * B`` first tries ``A.solve_left(B)``, then
        ``A.conjugate_transpose().i * B``.

    Note that vectors and matrices are treated quite differently,
    and that the only thing qualifying a matrix as a "vector" is
    its shape.

    We always assume that the right conversions etc. are present so
    that the expression can be computed in the fashion shown above.
    The output type is not selectable, it just becomes whatever it
    is.

    No in-place operations or buffer reuse is ever performed.
    """
    
    def __init__(self, add_graph=None, multiply_graph=None):
        if multiply_graph is None:
            from .core import multiply_graph
        if add_graph is None:
            from .core import addition_conversion_graph as add_graph
        self.multiply_graph = multiply_graph
        self.add_graph = add_graph

    def compile(self, matrix):
        operation_root = SimplisticCompilation(
            self.multiply_graph, self.add_graph).compile(matrix._expr)
        return operation_root

    def compute(self, matrix):
        return Matrix(self.compile(matrix).perform())
