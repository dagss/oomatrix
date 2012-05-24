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
from .task import Task, LeafTask
from .metadata import MatrixMetadata
from .cost_value import FLOP, INVOCATION


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

def get_cheapest_task(task_nodes, cost_map):
    task_nodes = list(task_nodes)
    min_cost = np.inf
    min_node = None
    for task_node in task_nodes:
        cost = task_node.task.total_cost.weigh(cost_map)
        if cost < min_cost:
            min_cost = cost
            min_node = task_node
    if min_node is None:
        raise ImpossibleOperationError('no possible computation')
    return min_node

def filter_results(generator, cost_map):
    # Sort by kind
    results_by_kind = {}
    for task_node in generator:
        lst = results_by_kind.setdefault(task_node.task.metadata.kind, [])
        lst.append(task_node)
    # Only take cheapest for each kind
    result = tuple(get_cheapest_task(lst, cost_map)
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

def _outer(partial_result, list_of_lists):
    current_list = list_of_lists[0]
    if len(list_of_lists) == 1:
        for x in current_list:
            yield partial_result + (x,)
    else:
        for x in current_list:
            for y in _outer(partial_result + (x,), list_of_lists[1:]):
                yield y

def outer(*lists):
    for x in _outer((), lists):
        yield x

class TaskNode:
    def __init__(self, task, conjugate_transpose):
        self.task = task
        self.is_conjugate_transpose = conjugate_transpose

    def tree(self):
        node = symbolic.Promise(self.task)
        if self.is_conjugate_transpose:
            node = symbolic.ConjugateTransposeNode(node)
        return node

    def conjugate_transpose_task(self):
        return TaskNode(self.task, not self.is_conjugate_transpose)        

def find_cost(computation, arg_tasks):
    assert all(isinstance(x, Task) for x in arg_tasks)
    if computation.cost is None:
        raise AssertionError('%s has no cost set' %
                             computation.name)
    metadata_list = [task.metadata for task in arg_tasks]            
    cost = computation.cost(*metadata_list) + INVOCATION
    if cost == 0:
        cost = cost_value.zero_cost
    if not isinstance(cost, cost_value.CostValue):
            raise TypeError('cost function %s for %s did not return 0 or a '
                            'CostValue' % (computation, computation.cost))
    return cost

class ExhaustiveCompilation(object):
    """
    Compiles an expression by trying all possibilities.

    Currently we just generate *all* in the slowest possible way ever,
    even without memoization, then take the cheapest. Of course,
    memoization should be added. Also, it should be possible refactor
    this into something like Dijkstra or A*.

    The input is a symbolic syntax tree, the output is a graph of Task
    objects.
    
    """

    def __init__(self):
        self.cache = {}
        self.cost_map = cost_value.default_cost_map

    def compile(self, node):
        gen = self.explore(node)
        return get_cheapest_task(gen, self.cost_map)

    # Visitor implementation
    def visit_multiply(self, node):
        return self.explore_multiplication(tuple(node.children))

    def visit_add(self, node):
        return self.explore_addition(tuple(node.children))

    def visit_leaf(self, node):
        metadata = MatrixMetadata(node.kind, (node.nrows,), (node.ncols,),
                                  node.dtype)
        task = LeafTask(node.matrix_impl, metadata, node)
        yield TaskNode(task, False)

    def visit_conjugate_transpose(self, node):
        # Recurse and process children, and then transpose the result
        child = node.child
        for task_node in child.accept_visitor(self, child):
            yield TaskNode(task_node.task,
                           not task_node.is_conjugate_transpose)
        # Also look at A.h -> B-style computations
        #rint node
        #or task_node in self.all_computations(node):
        #   print task_node
        #   yield task_node

    def visit_decomposition(self, node):
        from .decompositions import Factor
        child_metadata = MatrixMetadata(node.kind, (node.nrows,), (node.ncols,),
                                        node.dtype)
        decomposition = node.decomposition
        for child_task_node in node.child.accept_visitor(self, node.child):
            child_promise_tree = child_task_node.tree()
            tree = symbolic.DecompositionNode(child_promise_tree, decomposition)
            for decompose_task in self.all_computations(tree):
                yield decompose_task
        #child = node.child
        #for child_task_node in child.accept_visitor(self, child):
        #    child_task = child_task_node.task
        #    computation = decomposition.create_computation(metadata.kind)
        #    decompose_task = Task(computation, computation.cost(metadata),
        #                          [child_task], metadata, node)
        #    yield TaskNode(decompose_task, False)

    def visit_inverse(self, node):
        for child_task_node in node.child.accept_visitor(self, node.child):
            child_promise_tree = child_task_node.tree()
            tree = symbolic.InverseNode(child_promise_tree)
            for inverse_task in self.all_computations(tree):
                yield inverse_task

    def visit_bracket(self, node):
        # Compile child tree, and filter the resulting options with respect
        # to the target kinds requested here. Bracket nodes serves two
        # purposes: a) Enforcing specific matrix kinds as specific places
        # in the tree, b) ensure that the wrapped subtree doesn't participate
        # when using the distributive law, terms cancelling etc.
        child = node.child
        allowed_kinds = node.allowed_kinds
        for task_node in child.accept_visitor(self, child):
            if allowed_kinds is None:
                yield task_node
            else:
                if task_node.task.metadata.kind in allowed_kinds:
                    yield task_node
                else:
                    # post-operation conversion
                    for converted in self.generate_conversions_recursive(
                        task_node.tree(),
                        [task_node.task.metadata.kind],
                        only_kinds=allowed_kinds):
                        yield converted
            

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
        # The no-computation case
        if isinstance(expr, symbolic.LeafNode):
            assert False # (!!!)
            if (expr.kind not in avoid_kinds and
                expr.kind not in tried_kinds and
                (only_kinds is None or expr.kind in only_kinds)):
                yield expr
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
                arg_tasks = expr.as_computable_list(matched_key)
                metadata = MatrixMetadata(target_kind, (nrows,), (ncols,),
                                          dtype)
                cost = find_cost(computation, arg_tasks)
                task = Task(computation, cost, arg_tasks, metadata, expr)
                yield TaskNode(task, False)

    def generate_conversions(self, expr, tried_kinds, only_kinds=None):
        # Find all possible conversion computables that can be put on top
        # of the computable. Always pass a set of kinds already tried, to
        # avoid infinite loops
        for x in self.all_computations(expr, tried_kinds=tried_kinds,
                                       only_kinds=only_kinds):
            yield x

    def generate_conversions_recursive(self, expr, tried_kinds,
                                       only_kinds=None):
        # Find all possible *chains* of conversion computables that
        # can be put on top of the computable. I.e., like generate_conversions,
        # but allow more than one conversion.

        # For each possible conversion target...
        for x in self.generate_conversions(expr, tried_kinds):
            # Yield this conversion
            if only_kinds is None or x.task.metadata.kind in only_kinds:
                yield x
            # ...and recurse to add more conversions
            for y in self.generate_conversions_recursive(
                x.tree(),
                tried_kinds + [x.task.metadata.kind],
                only_kinds):
                yield y

    @memoizegenerator
    def explore_multiplication(self, operands):
        # TODO: Currently only explore pair-wise multiplication, will never
        # invoke A * B * C handlers and the like
        # EXCEPT for a special case of no conversions..., see below
        if len(operands) == 1:
            for x in self.explore(operands[0]):
                yield x
        #if len(operands) >= 2:
        #    # TODO this is to detect *exact* match of A * B * C, however
        #    # a three-term computation needing conversions won't be detected...
        #    expr = symbolic.MultiplyNode(operands)
        #    for x in self.all_computations(expr):
                yield x
        # Direct computations
        for split_idx in range(1, len(operands)):
            for x in self.explore_multiplication_split(operands, split_idx):
                assert isinstance(x, TaskNode)
                yield x
        # Look for opportunities to apply the distributive law
        for i, op in enumerate(operands):
            if op.can_distribute():
                for x in self.explore_distributive(operands[:i], op,
                                                   operands[i + 1:]):
                    yield x

    def explore_distributive(self, left_ops, op, right_ops):
        if len(left_ops) > 0:
            import warnings
            warnings.warn('need to implement right-distributive')
        if len(right_ops) > 0:
            right = symbolic.multiply(right_ops)
            new_node = op.distribute_right(right)
            new_node = symbolic.multiply(left_ops + (new_node,))
            for x in self.explore(new_node):
                yield x
            
                
    def explore_multiplication_split(self, operands, idx):
        left_task_nodes = self.explore_multiplication(operands[:idx])
        right_task_nodes = self.explore_multiplication(operands[idx:])
        for left, right in outer(left_task_nodes, right_task_nodes):
            for x in self.generate_pair_multiplications(
                        left, frozenset([left.task.metadata.kind]),
                        right, frozenset([right.task.metadata.kind])):
                yield x
            # Try to transpose back and forth
            for x in self.generate_pair_multiplications(
                        right.conjugate_transpose_task(),
                        frozenset([right.task.metadata.kind]),
                        left.conjugate_transpose_task(),
                        frozenset([left.task.metadata.kind])):
                yield x.conjugate_transpose_task()

    @memoizegenerator
    def generate_pair_multiplications(self,
                                      left, left_kinds_tried,
                                      right, right_kinds_tried):
        assert isinstance(left, TaskNode)
        assert isinstance(right, TaskNode)
        # Look up any direct computations
        expr = symbolic.MultiplyNode([left.tree(), right.tree()])
        for x in self.all_computations(expr):
            yield x
        # Recurse with all conversions of left operand
        for new_left in self.generate_conversions(left.tree(), left_kinds_tried):
            for x in self.generate_pair_multiplications(
                new_left,
                left_kinds_tried.union([new_left.task.metadata.kind]),
                right, right_kinds_tried):
                yield x
        # Recurse with all conversions of right operand
        for new_right in self.generate_conversions(right.tree(), right_kinds_tried):
            for x in self.generate_pair_multiplications(
                left,
                left_kinds_tried,
                new_right,
                right_kinds_tried.union([new_right.task.metadata.kind])):
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
            left_tasks = self.explore_addition(left_operands)
            right_tasks = self.explore_addition(right_operands)
            # right_computations is used multiple times, but the
            # @memoizegenerator ensures it is a tuple
            assert type(right_tasks) is tuple
            for left in left_tasks:
                for right in right_tasks:
                    for x in self.generate_pair_additions(
                        left, frozenset([left.task.metadata.kind]),
                        right, frozenset([right.task.metadata.kind])):
                        yield x
            
    def generate_pair_additions(self, left, left_kinds_tried,
                                right, right_kinds_tried):
        assert isinstance(left, TaskNode)
        assert isinstance(right, TaskNode)
        expr = symbolic.add([left.tree(), right.tree()])
        for x in self.all_computations(expr):
            yield x
        for new_left in self.generate_conversions(left.tree(), left_kinds_tried):
            for x in self.generate_pair_additions(
                new_left, left_kinds_tried.union([new_left.task.metadata.kind]),
                right, right_kinds_tried):
                yield x
        for new_right in self.generate_conversions(right.tree(), right_kinds_tried):
            for x in self.generate_pair_additions(
                left, left_kinds_tried,
                new_right, right_kinds_tried.union([new_right.task.metadata.kind])):
                yield x
                
        

class BaseCompiler(object):
    def __init__(self):
        self.cache = {}

    def compile(self, expression):
        key = expression.metadata_tree()
        result = self.cache.get(key, None)
        if result is None: 
            task_node = self.compilation_factory().compile(expression)
            result = (task_node.task, task_node.is_conjugate_transpose)
            self.cache[key] = result
        return result
class ExhaustiveCompiler(BaseCompiler):
    compilation_factory = ExhaustiveCompilation

exhaustive_compiler_instance = ExhaustiveCompiler()
