"""
Compilers take a syntactic-tree (symbolic.py), an abstract description
of the computation to be done, and turns it into a computable-tree
(computation.py), which essentially binds together chosen computation
routines and their arguments.
"""


import sys
import numpy as np
from itertools import izip, chain, combinations, permutations

# TODO: Computers should be reentrant/thread-safe, since they can
# be assigned to a global configuration variable.

from . import formatter, symbolic, actions
from .matrix import Matrix
from .operation_graphs import (
    ImpossibleOperationError,
    addition_graph,
    multiplication_graph)
from .computation import BaseComputable, ComputableLeaf, Computable
from .kind import lookup_computations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def set_of_pairwise_nonempty_splits(iterable):
    # TODO: Should eliminate subset vs. complement duplicates
    pool = tuple(iterable)
    n = len(pool)
    for r in range(1, n):
        for indices in permutations(range(n), r):
            if sorted(indices) == list(indices):
                subset = tuple(pool[i] for i in indices)
                complement = tuple(pool[i] for i in range(n) if i not in indices)
                yield subset, complement
            


class ExhaustiveCompilation(object):
    """
    Compiles an expression by trying all possibilities.

    Currently we just generate *all* in the slowest possible way ever,
    even without memoization, then take the cheapest. Of course,
    memoization should be added. Also, it should be possible refactor
    this into something like Dijkstra or A*.
    """

    # Each visitor method simply yields (cnode, ctranspose) for all
    # possibilities it can find, regardless of target
    # type. `ctranspose` is True if `cnode` computes the conjugate
    # transpose of the input.
    #
    # cnode: node in a computable-tree
    # snode: node in a syntactic-tree

    def compile(self, snode):
        possible_cnodes = list(self.explore(snode))
        possible_cnodes.sort(key=lambda cnode: cnode.cost)
        for cnode in possible_cnodes:
            #if cnode.kind in target_kinds:
            return cnode
        raise ImpossibleOperationError()

    # Visitor implementation
    def visit_multiply(self, snode):
        return self.explore_multiplication(snode.children)

    def visit_add(self, snode):
        return self.explore_addition(snode.children)

    def visit_leaf(self, snode):
        yield ComputableLeaf(snode.matrix_impl)

    #
    # Exploration
    # 
    # Naming convention: Methods that start with explore_ takes symbolic
    # node arguments (possibly as lists), while methods that start with
    # generate_ takes computables
    #

    def explore(self, snode):
        return snode.accept_visitor(self, snode)

    def all_computations(self, match_pattern, children, avoid_kinds=(),
                         only_kinds=None):
        nrows = children[0].nrows
        ncols = children[0].ncols
        dtype = children[0].dtype # todo
        # Generates all computables possible for the match_pattern
        computations_by_kind = lookup_computations(match_pattern)
        for target_kind, computations in computations_by_kind.iteritems():
            if target_kind in avoid_kinds:
                continue
            if only_kinds is not None and target_kind not in only_kinds:
                continue
            for computation in computations:
                yield Computable(computation, children, nrows, ncols, dtype)

    def generate_conversions(self, computable, avoid_kinds):
        # Find all possible conversion computables that can be put on top
        # of the computable. Always pass a set of kinds already tried, to
        # avoid infinite loops
        for x in self.all_computations(computable.kind, [computable],
                                       avoid_kinds=avoid_kinds):
            yield x

    def explore_multiplication(self, operands):
        # TODO: Currently only explore pair-wise multiplication, will never
        # invoke A * B * C handlers and the like
        if len(operands) == 1:
            for x in self.explore(operands[0]):
                yield x

        for split_idx in range(1, len(operands)):
            for x in self.explore_multiplication_split(operands, split_idx):
                yield x
                
    def explore_multiplication_split(self, operands, idx):
        left_computables = self.explore_multiplication(operands[:idx])
        right_computables = self.explore_multiplication(operands[idx:])
        right_computables = list(right_computables) # will reuse many times
        for left in left_computables:
            for right in right_computables:
                for x in self.generate_pair_multiplications(
                        left, [left.kind], right, [right.kind]):
                    yield x

    def generate_pair_multiplications(self,
                                      left, left_kinds_tried,
                                      right, right_kinds_tried):
        assert isinstance(left, BaseComputable)
        assert isinstance(right, BaseComputable)
        # Look up any direct computations
        for x in self.all_computations(left.kind * right.kind,
                                       [left, right]):
            yield x
        # Do all conversions of left operand
        for new_left in self.generate_conversions(left, left_kinds_tried):
            for x in self.generate_pair_multiplications(
                    new_left, left_kinds_tried + [new_left.kind],
                    right, right_kinds_tried):
                yield x
        # Do all conversions of right operand
        for new_right in self.generate_conversions(right, right_kinds_tried):
            for x in self.generate_pair_multiplications(
                    left, left_kinds_tried,
                    new_right, right_kinds_tried + [new_right.kind]):
                yield x

    def explore_addition(self, operands):
        # TODO: Currently only explores nargs =2-computations
        if len(operands) == 1:
            for x in self.explore(operands[0]):
                yield x

        # Try for an exact match (an addition computation that takes
        # len(operands) arguments)
        #for x in self.generate_additions(operands):
        #    yield x

        # Split into subsets and try again
        gen = set_of_pairwise_nonempty_splits(operands)
        for left_operands, right_operands in gen:
            # TODO: This is where we want to recursively split the complement,
            # and check for nargs>2-computations as well
            left_computations = self.explore_addition(left_operands)
            right_computations = self.explore_addition(right_operands)
            right_computations = list(right_computations) # need more than once
            for left in left_computations:
                for right in right_computations:
                    for x in self.generate_additions([left, right]):
                        yield x
            
    def generate_additions(self, operands):
        pattern = sum((operand.kind for operand in operands[1:]),
                      operands[0].kind)
        for x in self.all_computations(pattern, operands):
            yield x
        


def is_right_vector(expr):
    return expr.ncols == 1 and expr.nrows > 1

def is_left_vector(expr):
    return expr.nrows == 1 and expr.ncols > 1

class SimplisticCompilation(object):
    def __init__(self):
        pass

    def compile(self, symbolic_node, target_kinds=None):
        assert isinstance(symbolic_node, symbolic.ExpressionNode), (
            '%r is not symbolic node' % symbolic_node)
        computable = symbolic_node.accept_visitor(
            self, symbolic_node, target_kinds=target_kinds)
        assert isinstance(computable, BaseComputable)
        return computable

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
        node = multiplication_graph.find_cheapest_action(
            (left, right), target_kinds=target_kinds)
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
        return ComputableLeaf(expr.matrix_impl)
            
    def visit_inverse(self, expr, target_kinds):
        raise NotImplementedError()

    def visit_conjugate_transpose(self, expr, target_kinds): 
        if target_kinds is not None:
            raise NotImplementedError()
        child_action = self.compile(expr.child)
        return actions.ConjugateTransposeAction(child_action)

    def visit_bracket(self, expr, target_kinds):
        return self.compile(expr.child, target_kinds=expr.kinds)


class BaseCompiler(object):
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

    def compile(self, matrix):
        operation_root = self.compilation_factory().compile(matrix._expr)
        return operation_root

    def compute(self, matrix):
        return Matrix(self.compile(matrix).compute())

class SimplisticCompiler(BaseCompiler):
    compilation_factory = SimplisticCompilation

class ExhaustiveCompiler(BaseCompiler):
    compilation_factory = ExhaustiveCompilation

    def compile(self, matrix):
        operation_root = self.compilation_factory().compile(matrix._expr)
        return operation_root
