import sys
import numpy as np

# TODO: Computers should be reentrant/thread-safe, since they can
# be assigned to a global configuration variable.

from . import formatter, symbolic, actions
from .matrix import Matrix
from .core import ImpossibleOperationError

class SimplisticCompilation(object):
    def __init__(self, multiply_graph):
        self.multiply_graph = multiply_graph

    def is_right_vector(self, expr):
        return expr.ncols == 1 and expr.nrows > 1

    def is_left_vector(self, expr):
        return expr.nrows == 1 and expr.ncols > 1
    
    def compile(self, symbolic_node, target_kinds=None):
        assert isinstance(symbolic_node, symbolic.ExpressionNode), (
            '%r is not symbolic node' % symbolic_node)
        action_node = symbolic_node.accept_visitor(
            self, symbolic_node, target_kinds=target_kinds)
        assert isinstance(action_node, actions.Action)
        return action_node

    def visit_add(self, expr, target_kinds):
        raise NotImplementedError()

    def visit_multiply(self, expr, target_kinds):
        def mul_pair(left, right):
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
        
        assert len(expr.children) >= 2

        # Figure out computation of each child node before looking at
        # how to multiply them together
        children = [self.compile(x) for x in expr.children]

        
        # Figure out if this is a "matrix-vector" product, in which
        # case we change the order of multiplication.
        if self.is_right_vector(expr):
            # Right-to-left
            right = children[-1]
            for left in children[-2::-1]:
                right = mul_pair(left, right)
            return right
        else:
            # Left-to-right
            left = children[0]
            for right in children[1:]:
                left = mul_pair(left, right)
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
    
    def __init__(self, multiply_graph=None):
        if multiply_graph is None:
            from .core import multiply_graph
        self.multiply_graph = multiply_graph

    def compile(self, matrix):
        operation_root = SimplisticCompilation(
            self.multiply_graph).compile(matrix._expr)
        return operation_root

    def compute(self, matrix):
        return Matrix(self.compile(matrix).perform())
