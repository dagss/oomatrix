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

A neighbour in the graph is either a zero-cost edge corresponding to
a re-expression of the expression tree (use associative or distributive
rules), or the conversion of symbolic nodes to Tasks (or a conversion, i.e.
only the insertion of a Task). Each neighour should only represent a single
"atomic" change to the tree; i.e. you don't immedietaly keep
processing but emit a neighbour which can then be further processed
when it is visited.

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
from .task import Task
from .symbolic import TaskLeaf
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
            
def frozenset_union(*args):
    return frozenset().union(*args)

class NeighbourExpressionGraphGenerator(object):
    """
    Recurses through an expression graph in order to generate its
    neighbour graph. At each level one yields (subtree_root,
    task_set); the task_set is a set of all the tasks in the tree and
    is used to compute the cost (since tasks may have common
    dependencies, this is not simply the sum of all Task instances in
    the tree).

    self.process caches, so that when a neighbour is visited and its neighbours
    searched for, a lot of the existing nodes is used to represent that and
    only the actual changed path through the tree requires more objects and
    computation time

    

    
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
        key, universe = transforms.kind_key_transform(node)
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
        # The cache here is important, see class docstring
        results = self.cache.get(node, None)
        if results is None:
            results = []
            # Try for an exact computation implemented for this expression tree
            for x in self.generate_direct_computations(node):
                results.append(x)
            # Look for ways to process the subtree
            for x in node.accept_visitor(self, node):
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
                new_dependencies = frozenset_union(*[getattr(x, 'dependencies', ())
                                                     for x in new_children])
                new_node = symbolic.AddNode(new_children)
                new_node.dependencies = new_dependencies
                yield new_node

        # Use associative rules to split up expression
        for subset, complement in set_of_pairwise_nonempty_splits(children):
            left = symbolic.AddNode(subset) if len(subset) > 1 else subset[0]
            right = (symbolic.AddNode(complement)
                     if len(complement) > 1 else complement[0])
            new_parent = symbolic.AddNode(sorted([left, right]))
            yield new_parent

    def visit_multiply(self, node):
        # Forward possibilities in sub-trees. These include conversions,
        # so that all direction additions are already covered by process
        children = node.children
        for i, child in enumerate(children):
            for new_child in self.process(child):            
                new_children = children[:i] + [new_child] + children[i + 1:]
                new_dependencies = frozenset_union(*[getattr(x, 'dependencies', ())
                                                     for x in new_children])
                new_node = symbolic.MultiplyNode(new_children)
                new_node.dependencies = new_dependencies
                yield new_node

        # Use associative rules to split up expression
        for i in range(1, len(children) - 1):
            left_children, right_children = children[:i], children[i:]
            left_node = (symbolic.MultiplyNode(left_children)
                         if len(left_children) > 1 else left_children[0])
            right_node = (symbolic.MultiplyNode(right_children)
                          if len(right_children) > 1 else right_children[0])
            yield symbolic.MultiplyNode([left_node, right_node])

        if len(node.children) == 2:
            # Use distributive rule; it is enough to consider this case because
            # larger cases are eventually reduced to this one in all possible
            # ways
            left, right = node.children
            if left.can_distribute():
                yield left.distribute_right(right)
            if right.can_distribute():
                yield right.distribute_left(left)
             

    def visit_conjugate_transpose(self, node):
        for new_child in self.process(node.child):
            yield symbolic.ConjugateTransposeNode(new_child)

    def visit_inverse(self, node):
        for new_child in self.process(node.child):
            yield symbolic.InverseNode(new_child)
    
    def visit_decomposition(self, node):
        for new_child in self.process(node.child):
            yield symbolic.DecompositionNode(new_child, node.decomposition)
        
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
            if self.is_goal(head):
                return head # Done!
            
            gen = self.neighbour_generator.generate_neighbours(head)
            for cost, node in gen:
                cost_scalar = cost.weigh(self.cost_map)
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
        result = self.cache.get(meta_tree, None)
        if result is None:
            result = self.compilation_factory().compile(meta_tree)
            self.cache[meta_tree] = result
        return result, args

default_compiler_instance = ShortestPathCompilation()
