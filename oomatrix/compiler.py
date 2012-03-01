"""
Compilers take a tree consisting of only symbolic operations (arithmetic
etc.) and LeafNode's, and turns it into a tree consisting only of
ComputableNode and LeafNode, which described the

While searching for computation routines, the tree (fragments) in use
are hybrids of sorts, treating any BaseComputable as leaf nodes and stringing
them together with arithmetic operations to query for computation routines.
"""


import sys
import numpy as np
from itertools import izip, chain, combinations, permutations

# TODO: Computers should be reentrant/thread-safe, since they can
# be assigned to a global configuration variable.

from . import formatter, symbolic, cost_value
from .kind import lookup_computations, MatrixKind
from .computation import ImpossibleOperationError
from pprint import pprint

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
            

GRAY = object()

def get_cheapest_node(nodes, cost_map):
    min_cost = np.inf
    min_node = None
    for node in nodes:
        cost = node.cost.weigh(cost_map)
        if cost < min_cost:
            min_cost = cost
            min_node = node
    if min_node is None:
        raise ImpossibleOperationError('empty node list')
    return min_node

def filter_results(generator, cost_map):
    # Sort by kind
    results_by_kind = {}
    for node in generator:
        lst = results_by_kind.setdefault(node.kind, [])
        lst.append(node)
    # Only take cheapest for each kind
    result = tuple(get_cheapest_node(lst, cost_map)
                   for key, lst in results_by_kind.iteritems())
    return result

def memoizegenerator(method):
    """Used for decorating methods in ExhaustiveCompilation in order
    to use memoization.

    Also, filters the results so that only one result per kind is returned.
    """
    def new_method(self, *args):
        # no keyword args allowed
        key = (id(method), tuple(args))
        x = self.cache.get(key, None)
        if x is None:
            self.cache[key] = GRAY
            x = filter_results(method(self, *args), self.cost_map)
            self.cache[key] = x
        elif x is GRAY:
            raise AssertionError('Infinite loop')
        return x
    new_method.__name__ = method.__name__
    return new_method

class ExhaustiveCompilation(object):
    """
    Compiles an expression by trying all possibilities.

    Currently we just generate *all* in the slowest possible way ever,
    even without memoization, then take the cheapest. Of course,
    memoization should be added. Also, it should be possible refactor
    this into something like Dijkstra or A*.
    """
    _level = 0

    def __init__(self):
        self.cache = {}
        self.cost_map = cost_value.default_cost_map

    def compile(self, node):
        gen = self.explore(node)
        return get_cheapest_node(gen, self.cost_map)

    # Visitor implementation
    def visit_multiply(self, node):
        return self.explore_multiplication(tuple(node.children))

    def visit_add(self, node):
        return self.explore_addition(tuple(node.children))

    def visit_leaf(self, node):
        yield node

    def visit_conjugate_transpose(self, node):
        # Recurse and process children, and then transpose the result
        child = node.child
        for new_node in child.accept_visitor(self, child):
            yield symbolic.ConjugateTransposeNode(new_node)
        # Also look at A.h -> A-style computations
        #for new_node in self.all_computations(node):
        #    yield new_node
        # TODO: write test for above two lines and uncomment

    def visit_decomposition(self, node):
        # Compile child tree and leave the decomposition node intact
        child = node.child
        for new_node in child.accept_visitor(self, child):
            yield symbolic.DecompositionNode(new_node, node.decomposition)

    def visit_bracket(self, node):
        # Compile child tree, and filter the resulting options with respect
        # to the target kinds requested here. Bracket nodes serves two
        # purposes: a) Enforcing specific matrix kinds as specific places
        # in the tree, b) ensure that the wrapped subtree doesn't participate
        # when using the distributive law, terms cancelling etc.
        child = node.child
        allowed_kinds = node.allowed_kinds
        for new_node in child.accept_visitor(self, child):
            if allowed_kinds is None:
                yield new_node
            else:
                if new_node.kind in allowed_kinds:
                    yield new_node
                else:
                    # post-operation conversion
                    for converted_node in self.generate_conversions_recursive(
                        new_node, [new_node.kind], only_kinds=allowed_kinds):
                        yield converted_node
            

    #
    # Exploration
    # 
    # Naming convention: Methods that start with explore_ takes symbolic
    # node arguments (possibly as lists), while methods that start with
    # generate_ takes computables. process_... returns single nodes, are not
    # generators
    #

    def explore(self, snode):
        return snode.accept_visitor(self, snode)

    def all_computations(self, expr, avoid_kinds=(), only_kinds=None,
                         tried_kinds=()):
        # Looks up all *directly* matching computations
        #
        # tried_kinds is typically used to avoid infinite loops (and
        # can be done away with if we do memoization with "gray-marking").
        # It is like avoid_kinds in that it avoids going *to* any of the kinds,
        # but it only does so if expr is trivial (so that "kind.h -> kind" is
        # allowed).
        nrows = expr.nrows
        ncols = expr.ncols
        dtype = expr.dtype # todo
        # Generates all computables possible for the match_pattern
        key = expr.get_key()
        trivial = isinstance(key, MatrixKind)
        computations_by_kind = expr.universe.get_computations(key)
        for target_kind, computations in computations_by_kind.iteritems():
            if trivial and target_kind in tried_kinds:
                continue
            if target_kind in avoid_kinds:
                continue
            if only_kinds is not None and target_kind not in only_kinds:
                continue
            for computation in computations:
                matched_key = computation.match
                args = expr.as_computable_list(matched_key)
                yield symbolic.ComputableNode(computation, args, nrows,
                                              ncols, dtype, expr)

    def generate_conversions(self, computable, tried_kinds, only_kinds=None):
        # Find all possible conversion computables that can be put on top
        # of the computable. Always pass a set of kinds already tried, to
        # avoid infinite loops
        for x in self.all_computations(computable, tried_kinds=tried_kinds,
                                       only_kinds=only_kinds):
            yield x

    def generate_conversions_recursive(self, computable, tried_kinds,
                                       only_kinds=None):
        # Find all possible *chains* of conversion computables that
        # can be put on top of the computable. I.e., like generate_conversions,
        # but allow more than one conversion.

        # For each possible conversion target...
        for x in self.generate_conversions(computable, tried_kinds):
            # Yield this conversion
            if only_kinds is None or x.kind in only_kinds:
                yield x
            # ...and recurse to add more conversions
            for y in self.generate_conversions_recursive(x,
                                                         tried_kinds + [x.kind],
                                                         only_kinds):
                yield y

    @memoizegenerator
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
        self._level += 1
        left_computables = self.explore_multiplication(operands[:idx])
        right_computables = self.explore_multiplication(operands[idx:])
        self._level -= 1
        for left in left_computables:
            for right in right_computables:
                for x in self.generate_pair_multiplications(
                        left, frozenset([left.kind]),
                        right, frozenset([right.kind]), False):
                    yield x

    @memoizegenerator
    def generate_pair_multiplications(self,
                                      left, left_kinds_tried,
                                      right, right_kinds_tried,
                                      transpose_tried):
        for x in (left, right):
            assert isinstance(x, (symbolic.BaseComputable,
                                  symbolic.ConjugateTransposeNode))
        # Look up any direct computations
        expr = symbolic.MultiplyNode([left, right])
        for x in self.all_computations(expr):
            yield x
        # Look at conjugating back and forth
        if not transpose_tried:
            for x in self.generate_pair_multiplications(
                symbolic.ConjugateTransposeNode(right), frozenset([right.kind]),
                symbolic.ConjugateTransposeNode(left), frozenset([left.kind]),
                True):
                yield symbolic.ConjugateTransposeNode(x)
        # Recurse with all conversions of left operand
        for new_left in self.generate_conversions(left, left_kinds_tried):
            for x in self.generate_pair_multiplications(
                    new_left, left_kinds_tried.union([new_left.kind]),
                    right, right_kinds_tried,
                    transpose_tried):
                yield x
        # Recurse with all conversions of right operand
        for new_right in self.generate_conversions(right, right_kinds_tried):
            for x in self.generate_pair_multiplications(
                    left, left_kinds_tried,
                    new_right, right_kinds_tried.union([new_right.kind]),
                    transpose_tried):
                yield x

    @memoizegenerator
    def explore_addition(self, operands):
        # TODO: Currently only explores nargs =2-computations
        if len(operands) == 1:
            for x in self.explore(operands[0]):
                yield x

        # Split into all possible pairs
        gen = set_of_pairwise_nonempty_splits(operands)
        for left_operands, right_operands in gen:
            # TODO: This is where we want to recursively split the complement,
            # and check for nargs>2-computations as well
            left_computations = self.explore_addition(left_operands)
            right_computations = self.explore_addition(right_operands)
            # right_computations is used multiple times, but the
            # @memoizegenerator ensures it is a tuple
            assert type(right_computations) is tuple
            for left in left_computations:
                for right in right_computations:
                    for x in self.generate_pair_additions(
                        left, frozenset([left.kind]),
                        right, frozenset([right.kind])):
                        yield x
            
    def generate_pair_additions(self, left, left_kinds_tried,
                                right, right_kinds_tried):
        expr = symbolic.AddNode([left, right])
        for x in self.all_computations(expr):
            yield x
        for new_left in self.generate_conversions(left, left_kinds_tried):
            for x in self.generate_pair_additions(
                new_left, left_kinds_tried.union([new_left.kind]),
                right, right_kinds_tried):
                yield x
        for new_right in self.generate_conversions(right, right_kinds_tried):
            for x in self.generate_pair_additions(
                left, left_kinds_tried,
                new_right, right_kinds_tried.union([new_right.kind])):
                yield x
                
        

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

    def compile(self, expression):
        operation_root = self.compilation_factory().compile(expression)
        return operation_root

class ExhaustiveCompiler(BaseCompiler):
    compilation_factory = ExhaustiveCompilation

