"""
The input to a compilation is a DAG consisting of symbolic nodes
(AddNode, ConjugateTransposeNode, ...) and LeafNode wrapping
MatrixImpl instances.

Caching
-------

The compilers implements a cache by first separating the expression
graph into a metadata graph containing symbolic nodes and
MatrixMetadataLeaf nodes.  These contain i) metadata for the matrix
and ii) an integer identifying the actual matrix instance, but does
not actually contain the matrix content.  Then the cache is looked up
for a compiled expression that is"essentially"
the same (i.e. the same metadata, but matrix content can be different, since
MatrixImpl references have been replaced with integeres).

Compilation
-----------

Compiling a program for expression evaluation is essentially finding
the shortest path through a graph where each vertex is an expression
graph.  Each vertex is a DAG of symbolic nodes, MatrixMetadataLeaf,
and Task instances.

The start node contains no Task instances and is a purely symbolic tree.
A valid goal node contains no symbolic nodes and strings together the
expression only using Task instances.

NOTE: The trees are at all times kept sorted! (By the metadata)

(TODO make exception: It should be OK to have a single
ConjugateTransposeNode wrapping the root task.)

A neighbour in the tree is most of the time defined as the insertion of one
more Task (either eliminating symbolic nodes in the process, or doing a
conversion), though for convenience there are also some zero-cost edges
(like applying the distributive rule).

The shortest path is found by Dijkstra. Finding the neighbour-tree is done
by a visitor class which recurses through the tree and, eventually, generates
all the neighbour trees. Each "yield" statement deep in the tree eventually
bubbles up to the top; i.e. psuedo-code for an add statement would be::

    # Yield all neighbours we can get to by modifications in sub-trees 
    for each child_tree:
        for x in self.generate_neighbours(child_tree):
            yield current AddNode with child_tree replaced by x
    # Then, yield all possible ways of performing parts of the addition;
    # for each replacement of some of the children with a Task representing
    # their addition we yield that possibility


"""


import sys
import numpy as np
from itertools import izip, chain, combinations, permutations

# TODO: Computers should be reentrant/thread-safe, since they can
# be assigned to a global configuration variable.

from . import formatter, symbolic, cost_value, transforms
from .kind import lookup_computations, MatrixKind
from .computation import ImpossibleOperationError
from pprint import pprint
from .task import Task, LeafTask
from .metadata import MatrixMetadata
from .cost_value import FLOP, INVOCATION
from .heap import Heap

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def set_of_pairwise_nonempty_splits(iterable):
    pool = tuple(iterable)
    n = len(pool)
    for r in range(1, n // 2 + 1):
        for indices in permutations(range(n), r):
            if sorted(indices) == list(indices):
                subset = [pool[i] for i in indices]
                complement = [pool[i] for i in range(n) if i not in indices]
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

def frozenset_union(*args):
    return frozenset().union(*args)


class TaskLeaf(symbolic.ExpressionNode):
    kind = universe = ncols = nrows = dtype = None # TODO remove these from symbolic tree
    children = ()
    precedence = 1000
    
    def __init__(self, task):
        self.task = task
        self.metadata = task.metadata
        self.dependencies = task.dependencies

    def as_tuple(self):
        # TaskNode compares by its metadata first and then the task (which must
        # be different from the integer IDs used in MatrixMetadataNode).
        # This ensures that sorting happens by class first.
        return self.metadata.as_tuple() + ('task', id(self.task))

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_task_leaf(*args, **kw)

    def _repr(self, indent):
        return [indent + '<TaskLeaf %r>' % self.metadata]

class NeighbourExpressionGraphGenerator(object):
    """
    Recurses through an expression graph in order to generate its
    neighbour graph. At each level one yields (subtree_root,
    task_set); the task_set is a set of all the tasks in the tree and
    is used to compute the cost (since tasks may have common
    dependencies, this is not simply the sum of all Task instances in
    the tree).

    

    
    """
    def __init__(self):
        self.cache = {}
    
    def generate_neighbours(self, node):
        for subtree in self.process(node):
            tasks = getattr(subtree, 'dependencies', ())
            cost = sum([task.cost for task in tasks], cost_value.zero_cost)
            yield cost, subtree
        
    def generate_direct_computations(self, node):
        if isinstance(node, Task):
            return
        #print '?????',node
        key, universe = transforms.kind_key_transform(node)
        if isinstance(key, tuple) and len(key) == 4:
            print 'NODE', node
            print 'KEY', key
        computations_by_kind = universe.get_computations(key)

        for target_kind, computations in computations_by_kind.iteritems():
            for computation in computations:
                root_meta, args = transforms.flatten(node)
                root_meta = root_meta.copy_with_kind(target_kind)
                meta_args = [arg.metadata for arg in args]
                args = [arg.task if isinstance(arg, TaskLeaf) else arg
                        for arg in args]
                cost = find_cost(computation, meta_args)
                task = Task(computation, cost, args, root_meta, node)
                yield TaskLeaf(task)

    def process(self, node):
        results = self.cache.get(node, None)
        if results is None:
            results = []
            # Try for an exact computation implemented for this expression tree
            for x in self.generate_direct_computations(node):
                results.append(x)
            # Look for ways to process the subtree
            for x in node.accept_visitor(self, node):
                print 'RECV', x
                if (isinstance(x, symbolic.AddNode) and len(x.children) == 3
                    and sorted(x.children) != x.children):
                    print 'FAILING INPUT', node
                    print 'FAILING OUTPUT', x
                    1/0
                    
                results.append(x)
            self.cache[node] = results
        return results

    def visit_add(self, node):
        # Forward possibilities in sub-trees. These include conversions,
        # so that all direction additions are already covered by process
        children = node.children
        for i, child in enumerate(children):
            for new_child in self.process(child):
                new_children = children[:i] + children[i + 1:]
                new_children.append(new_child)
                # Important: keep the children sorted (by kind)
                new_children.sort()
                if not any(isinstance(x, symbolic.AddNode) for x in new_children):
#                    if len(new_children) == 3:
                        print 'SORTED', new_children
                #if len(new_children) == 2:
                #    1/0
                new_dependencies = frozenset_union(*[getattr(x, 'dependencies', ())
                                                     for x in new_children])
                new_node = symbolic.AddNode(new_children)
                new_node.dependencies = new_dependencies
                print 'YIELDING', new_children
                yield new_node

        # Use associative rules to split up expression
        for subset, complement in set_of_pairwise_nonempty_splits(children):
            left = symbolic.AddNode(subset) if len(subset) > 1 else subset[0]
            right = (symbolic.AddNode(complement)
                     if len(complement) > 1 else complement[0])
            new_parent = symbolic.AddNode(sorted([left, right]))
            print 'YIELDING_B', new_parent
            yield new_parent
        
    def visit_metadata_leaf(self, node):
        # self.process has already covered all conversions of the leaf-node,
        # so we can't do anything here to produce a neighbour tree
        return ()

    visit_task_leaf = visit_metadata_leaf

class ShortestPathCompilation(object):
    def __init__(self):
        self.neighbour_generator = NeighbourExpressionGraphGenerator()
        self.cost_map = cost_value.default_cost_map
        
    def compile(self, root):
        # Use Dijkstra's algorithm, but we don't need the actual path, just
        # the resulting computation DAG
        #
        # Since Heap() doesn't have decrease-key, we instead use a visited
        # set to flag the nodes that have been taken out of the queue.
        visited = set()
        tentative_queue = Heap()
        tentative_queue.push(0, root)
        while len(tentative_queue) > 0:
            _, head = tentative_queue.pop()
            if head in visited:
                continue # visited earlier with a lower cost
            visited.add(head)
            print 'popped',head
            if self.is_goal(head):
                return head # Done!
            
            gen = self.neighbour_generator.generate_neighbours(head)
            for cost, node in gen:
                cost_scalar = cost.weigh(self.cost_map)
                print 'pushed'
                print node
                tentative_queue.push(cost_scalar, node)
        raise ImpossibleOperationError()

    def is_goal(self, node):
        return (isinstance(node, TaskLeaf) or
                (isinstance(node, symbolic.ConjugateTransposeNode) and
                 isinstance(node.child, TaskLeaf)))

def find_cost(computation, meta_args):
    assert all(isinstance(x, MatrixMetadata) for x in meta_args)
    if computation.cost is None:
        raise AssertionError('%s has no cost set' %
                             computation.name)
    cost = computation.cost(*meta_args) + INVOCATION
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

    def visit_metadata_leaf(self, node):
        task = LeafTask(node.leaf_index, node.metadata, node)
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
        key = transforms.kind_key_transform(expr)
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
        meta_tree, args = transforms.metadata_transform(expression)
        result = self.cache.get(meta_tree, None)
        if result is None or True: # TODO: Tasks must have switchable args
            task_node = self.compilation_factory().compile(meta_tree)
            result = (task_node.task, task_node.is_conjugate_transpose)
            self.cache[key] = result
        return result


class ShortestPathCompiler(BaseCompiler):
    compilation_factory = ShortestPathCompilation

    def compile(self, expression):
        meta_tree, args = transforms.metadata_transform(expression)
        #result = self.cache.get(meta_tree, None)
        #if result is None or True: # TODO: Tasks must have switchable args
        result = self.compilation_factory().compile(meta_tree)
        #    result = (task_node.task, task_node.is_conjugate_transpose)
        #    self.cache[key] = result
        return result, args


class ExhaustiveCompiler(BaseCompiler):
    compilation_factory = ExhaustiveCompilation

exhaustive_compiler_instance = ExhaustiveCompiler()
