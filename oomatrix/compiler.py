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

import os
import sys
import numpy as np
from itertools import izip, chain, combinations, permutations

do_trace = bool(int(os.environ.get("T", '0')))

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

def is_leaf(node):
    return (isinstance(node, (symbolic.MatrixMetadataLeaf, TaskLeaf)) or
            (isinstance(node, (symbolic.ConjugateTransposeNode, symbolic.InverseNode)) and
             isinstance(node.children[0], (symbolic.MatrixMetadataLeaf, TaskLeaf))))

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
            cost = sum([task.cost for task in subtree.task_dependencies],
                       cost_value.zero_cost)
            yield cost, subtree
        
    def generate_direct_computations(self, node):
        # Important: This spawns new Task objects, so important to
        # only call from process() so that each task is cached
        if isinstance(node, Task):
            assert False
        key, universe = transforms.kind_key_transform(node)
        computations_by_kind = universe.get_computations(key)
        root_meta, args = transforms.flatten(node)
        meta_args = [arg.metadata for arg in args]
        argument_index_set = frozenset_union(*[arg.argument_index_set
                                               for arg in args])
        args = [arg.as_task() for arg in args]

        for target_kind, computations in computations_by_kind.iteritems():
            root_meta = root_meta.copy_with_kind(target_kind)
            for computation in computations:
                cost = find_cost(computation, meta_args)
                task = Task(computation, cost, args, root_meta, node)
                new_node = TaskLeaf(task, argument_index_set)
                new_node.task_dependencies = node.task_dependencies.union([task])
                yield new_node

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
        else:
            pass
        return results

    def compile_add_children(self, node):
        children = node.children
        new_children = []
        for i, child in enumerate(children):
            if is_leaf(child):
                # todo: this will never explicitly transpose a matrix...
                new_children.append(child)
            else:
                sub_compiler = ShortestPathCompilation(self)
                if do_trace:
                    print '>>> SUBCOMPILATION START %x' % id(sub_compiler)
                try:
                    new_child = sub_compiler.compile(child)
                    if do_trace:
                        print '<<< SUBCOMPILATION SUCCESS %x' % id(sub_compiler)
                except ImpossibleOperationError:
                    if do_trace:
                        print '<<< SUBCOMPILATION FAILED %x' % id(sub_compiler)
                    return # yield no neighbours
                new_children.append(new_child)
        new_dependencies = frozenset_union(*[x.task_dependencies
                                             for x in new_children])
        new_node = symbolic.AddNode(new_children)
        new_node.task_dependencies = new_dependencies
        yield new_node        

    def process_add_associative(self, node):
        def add(children):
            if len(children) == 1:
                return children[0]
            else:
                node = symbolic.AddNode(children)
                node.task_dependencies  = frozenset_union(*[x.task_dependencies
                                                            for x in children])
                return node
            
        for subset, complement in set_of_pairwise_nonempty_splits(node.children):
            left = add(subset)
            right = add(complement)
            new_children = sorted([left, right])
            new_dependencies = frozenset_union(*[x.task_dependencies
                                                 for x in new_children])
            new_parent = symbolic.AddNode(new_children)
            new_parent.task_dependencies = new_dependencies
            yield new_parent

    def visit_add(self, node):
        # At this point, a direction match of +-ing the children has
        # already been attempted by self.process()
        children = node.children
        if not all(is_leaf(child) for child in children):
            # Some children are not fully computed; we turn greedy and
            # launch a sub-compilation for each of the children
            for x in self.compile_add_children(node):
                yield x
        else:
            # All children are fully computed; we a) use the associative rule to
            # split up the expression
            for x in self.process_add_associative(node):
                yield x

            def add(children):
                node = symbolic.AddNode(children)
                node.task_dependencies  = frozenset_union(*[x.task_dependencies
                                                            for x in children])
                return node
            
            # ...and b) when we're down to a pair, look
            # at possible conversions. (This is a mediocre solution...)
            if len(children) == 2:
                left, right = children
                for new_left in self.process(left):
                    new_children = sorted([new_left, right])
                    yield add(new_children)
                for new_right in self.process(right):
                    new_children = sorted([left, new_right])
                    yield add(new_children)                    


    def visit_multiply(self, node):
        # Forward possibilities in sub-trees. These include conversions.
        # We only do one sub-tree at the time, i.e. all left siblings should
        # be leaves when processing a sub-tree
        children = node.children
        for i, child in enumerate(children):
            did_process = False
            for new_child in self.process(child):
                did_process = True
                new_children = children[:i] + [new_child] + children[i + 1:]
                new_dependencies = frozenset_union(*[x.task_dependencies
                                                     for x in new_children])
                new_node = symbolic.MultiplyNode(new_children)
                new_node.task_dependencies = new_dependencies
                yield new_node
            if did_process:
                break

        # Use distributive rule; it is enough to consider the n=2 case
        # because larger cases are eventually reduced to this one in
        # all possible ways
        if len(node.children) == 2:
            left, right = node.children
            new_dependencies = left.task_dependencies.union(right.task_dependencies)
            if left.can_distribute():
                new_node = left.distribute_right(right)
                new_node.task_dependencies = new_dependencies
                yield new_node
            if right.can_distribute():
                new_node = right.distribute_left(left)
                new_node.task_dependencies = new_dependencies
                yield new_node

        # Use associative rules to split up expression
        def multiply(children):
            if len(children) == 1:
                return children[0]
            else:
                node = symbolic.MultiplyNode(children)
                node.task_dependencies  = frozenset_union(*[x.task_dependencies
                                                            for x in children])
                return node
        
        for i in range(1, len(children)):
            left_children, right_children = children[:i], children[i:]
            left_node = multiply(left_children)
            right_node = multiply(right_children)
            yield multiply([left_node, right_node])

        # Try the conjugate transpose expression
        if all(is_leaf(child) for child in children):
            new_children = [symbolic.conjugate_transpose(x) for x in children[::-1]]
            new_node = symbolic.MultiplyNode(new_children)
            new_node.task_dependencies = node.task_dependencies
            new_node_t = symbolic.ConjugateTransposeNode(new_node)
            new_node_t.task_dependencies = node.task_dependencies
            yield new_node_t


    def visit_conjugate_transpose(self, node):
        for new_child in self.process(node.child):
            new_node = symbolic.conjugate_transpose(new_child)
            new_node.task_dependencies = new_child.task_dependencies
            yield new_node

    def visit_inverse(self, node):
        for new_child in self.process(node.child):
            new_node = symbolic.inverse(new_child)
            new_node.task_dependencies = new_child.task_dependencies
            yield new_node
    
    def visit_decomposition(self, node):
        for new_child in self.process(node.child):
            new_node = symbolic.DecompositionNode(new_child, node.decomposition)
            new_node.task_dependencies = new_child.task_dependencies
            yield new_node
        
    def visit_metadata_leaf(self, node):
        # self.process has already covered all conversions of the leaf-node,
        # so we can't do anything here to produce a neighbour tree
        return ()



    visit_task_leaf = visit_metadata_leaf

class ShortestPathCompilation(object):
    def __init__(self, neighbour_generator=None):
        if neighbour_generator is None:
            neighbour_generator = NeighbourExpressionGraphGenerator()
        self.neighbour_generator = neighbour_generator
        self.cost_map = cost_value.default_cost_map
        
    def compile(self, root):
        # Use Dijkstra's algorithm, but we don't need the actual path, just
        # the resulting computation DAG
        #
        # Since Heap() doesn't have decrease-key, we instead use a visited
        # set to flag the nodes that have been taken out of the queue.
        #
        # Note: Each DAG (vertex in the graph) both represents a meta-data-tree,
        # *and* represents a specific way of computation. For comparisons here,
        # Task's are compared by their metadata and treated as a leaf. This
        # means that there can be multiple DAG for a vertex that compares
        # and hashes the same, but represents different ways and has different
        # costs. Since at the time of popping we pop the cheapest way to
        # the target, this is OK.
        #
        visited = set()
        tentative_queue = Heap()
        tentative_queue.push(0, root)
        vertex_count = 0
        edge_count = 0
        while len(tentative_queue) > 0:
            head_cost, head = tentative_queue.pop()
            edge_count += 1
            if do_trace and edge_count % 1000 == 0:
                print '|E|,|V|:', edge_count, vertex_count
            if head in visited:
                continue # visited earlier with a lower cost
            visited.add(head)
            vertex_count += 1

            if do_trace:
                print 'POP %x' % id(self), head
            if self.is_goal(head):
                if do_trace:
                    print 'GOAL REACHED, |E|, |V|:', edge_count, vertex_count
                return head # Done!
            
            gen = self.neighbour_generator.generate_neighbours(head)
            for cost, node in gen:
                cost_scalar = cost.weigh(self.cost_map)
                if do_trace:
                    print 'PUSH %x' % id(self), node
                tentative_queue.push(cost_scalar, node)
        raise ImpossibleOperationError()

    def is_goal(self, node):
        return (isinstance(node, TaskLeaf) or
                (isinstance(node, symbolic.ConjugateTransposeNode) and
                 isinstance(node.child, TaskLeaf)))

def find_cost(computation, meta_args):
    assert all(isinstance(x, MatrixMetadata) for x in meta_args)
    return computation.get_cost(meta_args) + INVOCATION

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
