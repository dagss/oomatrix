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

from . import formatter, symbolic, cost_value, transforms, utils, metadata, decompositions
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
    # todo: may be able to trim off last half of last chunk when n is even
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



class CompiledNode(object):
    """
    Objects of this class make up the final "program tree".

    It is immutable and compares by value (TBD).
    
    `children` are other CompiledNode instances whose output should
    be fed into `computation`.
    """
    def __init__(self, computation, weighted_cost, children, metadata, shuffle=None,
                 flat_shuffle=None):
        if shuffle is not None and flat_shuffle is not None:
            raise ValueError('either shuffle or flat_shuffle must be given')
        self.computation = computation
        self.weighted_cost = float(weighted_cost)
        self.children = tuple(children)
        self.metadata = metadata
        self.is_leaf = computation is None
        if self.is_leaf:
            if shuffle not in (None, ()):
                raise ValueError('invalid shuffle for leaf node')
            self.shuffle = ()
            self.arg_count = 1
        else:
            if shuffle is None:
                if flat_shuffle is None:
                    flat_shuffle = range(sum(child.arg_count for child in children))
                elif not all(isinstance(i, int) for i in flat_shuffle):
                    raise ValueError('invalid flat_shuffle')
                shuffle = []
                i = 0
                for child in self.children:
                    n = child.arg_count
                    shuffle.append(tuple(flat_shuffle[i:i + n]))
                    i += n
                self.shuffle = tuple(shuffle)
            else:
                self.shuffle = tuple(shuffle)
                valid_shuffle = len(self.shuffle) == len(self.children)
                for child, indices in zip(self.children, self.shuffle):
                    valid_shuffle = valid_shuffle and child.arg_count == len(indices)
                if not valid_shuffle:
                    raise ValueError('Invalid shuffle')
            flat_shuffle = sum(self.shuffle, ())
            self.arg_count = max(flat_shuffle) + 1
        # total_cost is computed
        self.total_cost = self.weighted_cost + sum(child.total_cost for child in self.children)

    def __eq__(self, other):
        # Note that this definition is recursive, as the comparison of children will
        # end up doing an element-wise comparison
        if type(other) is not CompiledNode:
            return False
        return (self.computation == other.computation and
                self.weighted_cost == other.weighted_cost and
                self.metadata == other.metadata and
                self.children == other.children and
                self.shuffle == other.shuffle)

    def __hash__(self):
        return hash((id(self.computation),
                     self.weighted_cost,
                     self.metadata,
                     self.children,
                     self.shuffle))

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def create_leaf(metadata):
        return CompiledNode(None, 0, (), metadata, ())

    def __repr__(self):
        lines = []
        self._repr('', lines)
        return '\n'.join(lines)

    def _repr(self, indent, lines):
        if self.is_leaf:
            lines.append(indent + '<leaf:%s cost=%.1f>' % (
                self.metadata.kind.name, self.weighted_cost))
        else:
            lines.append(indent + '<node:%s:%s cost=%.1f shuffle=%s;' % (
                self.metadata.kind.name,
                self.computation.name,
                self.total_cost,
                self.shuffle))
            for child in self.children:
                child._repr(indent + '  ', lines)
            lines.append(indent + '>')

    def leaves(self):
        if self.is_leaf:
            return [self]
        else:
            return sum([child.leaves() for child in self.children], [])

    def convert_to_task_graph(self, args):
        # Do a substitution, but use a "cache" dictionary `d` in the
        # node factory so that tasks doing exactly the same thing
        # results in the same object (by id()).
        d = {}
        def node_factory(node, new_children):
            new_children = tuple(new_children)
            task = d.get((node, new_children), None)
            if task is None:
                meta_args = [child.metadata for child in new_children]
                unweighted_cost = node.computation.get_cost(meta_args)
                task = Task(node.computation, unweighted_cost, new_children,
                            node.metadata, None)
                d[(node, new_children)] = task
            return task
        result = self.substitute(args, node_factory=node_factory)
        return result

    def substitute(self, args, shuffle=None, flat_shuffle=None, node_factory=None):
        """Substitute each leaf node of the tree rooted at `self` with the
        CompiledNodes given in args, and return the resulting tree.

        args can either be a list with `self.arg_count` elements (None meaning
        "do not replace"), or a dict mapping argument indices to replacement leaves.

        Optionally also converts the interior nodes in the tree by using a supplied
        node factory for the interior nodes.
        """
        if isinstance(args, dict):
            args, args_dict = [None] * self.arg_count, args
            for i, arg in args_dict.iteritems():
                args[i] = arg
        
        if node_factory is None:
            def node_factory(node, converted_children):
                return CompiledNode(node.computation, node.weighted_cost,
                                    converted_children, node.metadata,
                                    shuffle if node is self else None,
                                    flat_shuffle if node is self else None)

        return self._substitute(args, node_factory)

    def _substitute(self, args, node_factory):
        if self.is_leaf:
            r = args[0] or self
            assert r.metadata == self.metadata
            return r
        else:
            # Shuffle arguments and recurse
            new_children = []
            for child, child_shuffle in zip(self.children, self.shuffle):
                new_args = [args[i] for i in child_shuffle]
                new_child = child._substitute(new_args, node_factory)
                new_children.append(new_child)
            new_node = node_factory(self, new_children)
            return new_node
            
    def substitute_linked(self, indices, arg):
        return  self._substitute_linked(0, indices, arg)

    def _substitute_linked(self, offset, indices, arg):
        if self.is_leaf:
            r = arg if offset in indices else self
            assert r.metadata == self.metadata
            return r
        else:
            new_children = []
            for child, child_shuffle in zip(self.children, self.shuffle):
                new_child = child._substitute_linked(offset, indices, arg)
                new_children.append(new_child)
                offset += child.arg_count # note: of the old child
            new_node = CompiledNode(self.computation, self.weighted_cost, new_children,
                                    self.metadata)
            return new_node
            
            


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
        #print 'ENTERING GENERATE'
        for subtree in self.process(node):
            #print 'another subtree'
            cost = sum([task.cost for task in subtree.task_dependencies],
                       cost_value.zero_cost)
            yield cost, subtree
        #print 'LEAVING GENERATE'
        
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
                sub_compiler = DepthFirstCompilation(self)
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

class DepthFirstCompilation(object):
    def __init__(self, neighbour_generator=None):
        if neighbour_generator is None:
            neighbour_generator = NeighbourExpressionGraphGenerator()
        self.neighbour_generator = neighbour_generator
        self.cost_map = cost_value.default_cost_map
        
    def compile(self, root):
        self.node_count = 0
        self.upper_bound = np.inf
        self.solutions = []
        self.visited = set()
        self.explore(root)
        self.solutions.sort()
        if len(self.solutions) == 0:
            raise ImpossibleOperationError()
        else:
            return self.solutions[0][1]

    def explore(self, node):
        #print node
        self.node_count += 1
        if self.node_count % 100 == 0:
            print self.node_count
        self.visited.add(node)
        #1/0
        gen = self.neighbour_generator.generate_neighbours(node)
        for cost, child in gen:
            #print 'new child'
            if child in self.visited:
                continue
            cost_scalar = cost.weigh(self.cost_map)
            if cost_scalar > self.upper_bound:
                # prune branch
                continue
            if self.is_goal(child):
                if cost_scalar < self.upper_bound:
                    self.upper_bound = cost_scalar
                self.solutions.append((cost_scalar, child))
            self.explore(child)

    def is_goal(self, node):
        return (isinstance(node, TaskLeaf) or
                (isinstance(node, symbolic.ConjugateTransposeNode) and
                 isinstance(node.child, TaskLeaf)))


class RightToLeftCompilation(object):
    def __init__(self):
        self.cost_map = cost_value.default_cost_map

    def compile(self, root):
        return self.visit(root)

    def visit(self, node):
        result = node.accept_visitor(self, node)
        assert result is not None
        return result

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

    def find_cheapest_direct_computation(self, node):
        min_cost = np.inf
        best_task_node = None
        for task_node in self.generate_direct_computations(node):
            cost = sum([task.cost for task in task_node.task_dependencies],
                       cost_value.zero_cost)
            cost_scalar = cost.weigh(self.cost_map)
            if cost_scalar < min_cost:
                min_cost = cost_scalar
                best_task_node = task_node
        if best_task_node is None:
            raise ImpossibleOperationError()
        return best_task_node

    def visit_multiply(self, node):
        assert len(node.children) >= 2
        try:
            return self.find_cheapest_direct_computation(node)
        except ImpossibleOperationError:
            pass
        if len(node.children) > 2:
            # Always parenthize right-to-left
            right = symbolic.multiply(node.children[-2:])
            x = self.visit(right)
            y = self.visit(symbolic.multiply(node.children[:-2] + [x]))
            return y
        else:
            left, right = node.children
            if isinstance(right, symbolic.AddNode):
                # Allow summing up the right vector
                right_task_node = self.visit(right)
                new_node = symbolic.multiply([left, right_task_node])
                return self.visit(new_node)
            elif left.can_distribute():
                # Allow using the right-distributive law
                new_node = left.distribute_right(right)
                return self.visit(new_node)
            elif isinstance(left, symbolic.ConjugateTransposeNode):
                # Allow using A.h->B rules
                new_left = self.find_cheapest_direct_computation(left)
                new_node = symbolic.multiply([new_left, right])
                return self.find_cheapest_direct_computation(new_node)
        raise ImpossibleOperationError()

    def visit_add(self, node):
        try:
            return self.find_cheapest_direct_computation(node)
        except ImpossibleOperationError:
            pass
        # Always parenthize right-to-left
        if len(node.children) > 2:
            # Always parenthize right-to-left
            left = symbolic.add(node.children[:-2])
            right = symbolic.add(node.children[-2:])
            x = self.visit(right)
            y = self.visit(symbolic.add([left, x]))
            return y
        else:
            # Always fully compute all children
            left_node, right_node = node.children
            left_task = self.visit(left_node)
            right_task = self.visit(right_node)
            query_tree = symbolic.AddNode([left_task, right_task])
            return self.find_cheapest_direct_computation(query_tree)
        raise ImpossibleOperationError()

    def visit_task_leaf(self, node):
        return node



class ConversionCache(object):
    def __init__(self, cost_map):
        self._conversions = {} # { source_metadata : { target_metadata : (cost, conversion_obj_list) } } }
        self.cost_map = cost_map

    def _dfs(self, d, target_metadata, cost, conversion_list):
        old_cost, _ = d.get(target_metadata, (np.inf, None))
        if cost >= old_cost:
            return
        
        # Insert new, cheaper task...
        d[target_metadata] = (cost, conversion_list)
        # ...and recursively try all conversions from here to update kinds reachable from here
        conversions = get_cheapest_computations_by_metadata(target_metadata.kind.universe,
                                                            target_metadata.kind,
                                                            [target_metadata], self.cost_map)
        for next_target_kind, (next_cost_scalar, next_cost, next_conversion_obj) in (
            conversions.iteritems()):
            next_target_metadata = target_metadata.copy_with_kind(next_target_kind)
            next_conversion_list = conversion_list + [next_conversion_obj]
            self._dfs(d, next_target_metadata, next_cost_scalar + cost, next_conversion_list)
            
    def get_conversions_from(self, source_metadata):
        """
        Returns { kind : (cost_scalar, [computation]) }, with the cheapest
        way of converting `source_metadata` to every other reachable kind.
        `[computation]` is a list of conversion computation to be applied
        to the input operand, in order.
        """
        d = self._conversions.get(source_metadata)
        if d is None:
            self._conversions[source_metadata] = d = {}
            self._dfs(d, source_metadata, 0, [])
        return d

class ComputationCache(object):
    """
    For every (left_kind, right_kind), string together one (1!) computation
    and many conversions in order to figure out a way to perform the computation
    in the cheapest way under a given cost map. (Combining this to string together
    multiple computation is the job of the user of this component.)

    Overridden in AdditionCache and MultiplicationCache.
    """
    commutative_computation = False

    def __init__(self, conversion_cache):
        self.conversion_cache = conversion_cache
        self.cost_map = conversion_cache.cost_map
        self._computations = {} # { (op_kind, op_kind, ...) :
                                #   { result_kind : (cost_scalar, permutation, computation, [op_conversions],
                                #   [op_conversions], ...) } }

    def get_computations(self, metadata_list):
        if self.commutative_computation:
            kinds = [x.kind for x in metadata_list]
            if sorted(kinds) != kinds:
                raise ValueError("must have sorted operands")
        key = tuple(metadata_list)
        x = self._computations.get(key)
        if x is None:
            x = self._find_computation(key)
            self._computations[key] = x
        return x

class AdditionCache(ComputationCache):
    commutative_computation = True

    def _find_computation(self, metadata_list):
        # For now, assume that the number of operands is 2
        if len(metadata_list) != 2:
            raise NotImplementedError()
        left_meta, right_meta = metadata_list
        
        ld = self.conversion_cache.get_conversions_from(left_meta)
        rd = self.conversion_cache.get_conversions_from(right_meta)

        best = {} # { kind : (cost_scalar, permutation, adder, left_conv_list, right_conv_list) }
        for new_left_meta, (left_cost, left_ops) in ld.iteritems():
            for new_right_meta, (right_cost, right_ops) in rd.iteritems():
                lmeta, rmeta = new_left_meta, new_right_meta # avoid overwriting iteration vars
                reverse_order = lmeta.kind > rmeta.kind
                if reverse_order:
                    lmeta, rmeta = rmeta, lmeta
                d = get_cheapest_computations_by_metadata(
                    lmeta.kind.universe, lmeta.kind + rmeta.kind,
                    [lmeta, rmeta], self.cost_map)

                for target_kind, (adder_cost, _, adder) in d.iteritems():
                    old_result = best.get(target_kind, (np.inf,))
                    total_cost = left_cost + right_cost + adder_cost
                    if total_cost < old_result[0]:
                        p = (1, 0) if reverse_order else (0, 1)
                        best[target_kind] = (total_cost, p, adder, left_ops, right_ops)
        return best
    

def get_cheapest_computations_by_metadata(universe, match_expr, arg_metadatas, cost_map):
    """
    Returns the cheapest computation for each target kind, as
    a dict { target_kind : (cost_scalar, computation_object) }.

    match_expr: Kind-expression to match ("kind", "kind_a + kind_b", etc.)
    arg_metadatas: List of metadatas of each task
    target_meta: Kind-less MatrixMetadata.
    
    """
    key = match_expr.get_key()
    computations_by_kind = universe.get_computations(key)
    result = {}
    for target_kind, computations in computations_by_kind.iteritems():
        # For each kind, we pick out the cheapest computation
        costs = [comp.get_cost(arg_metadatas) for comp in computations]
        cost_scalars = [cost.weigh(cost_map) for cost in costs]
        i = np.argmin(cost_scalars)
        result[target_kind] = (cost_scalars[i], costs[i], computations[i])
    return result

def get_cheapest_computations(universe, match_expr, args, target_metadata, cost_map, symbolic_expr):
    """
    Returns the cheapest computation for each target kind, as
    a list of Task.

    match_expr: Kind-expression to match ("kind", "kind_a + kind_b", etc.)
    args: List of Task arguments to the computation
    target_meta: The metadata of the result, *except* that the kind is ignored/overwritten. This is
      used as a template when constructing tasks. (TBD: refactor)
    
    """
    arg_metadatas = [arg.metadata for arg in args]
    d = get_cheapest_computations_by_metadata(universe, match_expr, arg_metadatas,
                                              cost_map)
    possible_tasks = []
    for target_kind, (cost_scalar, cost, computation) in d.iteritems():
        metadata = target_metadata.copy_with_kind(target_kind)
        task = Task(computation, cost, args, metadata, symbolic_expr)
        possible_tasks.append(task)
    return possible_tasks

def fill_in_conversions(options, cost_map):
    """
    The given task_options are supposed to be different possible
    tasks for the *same* computation. Complete this list by adding
    conversions to all possible kinds. The resulting list will contain
    exactly one option for each kind (the cheapest one), and will be
    sorted from cheapest to most expensive.
    """
    # For each option task, run a depth first search through all conversions
    def dfs_insert(d, task):
        meta = task.metadata
        kind = meta.kind
        # Abort if it is more expensive than existing task
        old_cost, old_task = d.get(kind, (np.inf, None))
        cost = task.get_total_cost().weigh(cost_map)
        if cost >= old_cost:
            return

        # Insert new, cheaper task...
        d[kind] = (cost, task)

        # ...and recursively try all conversions from here
        new_tasks = get_cheapest_computations(kind.universe, kind, [task], meta, cost_map, None)
        for new_task in new_tasks:
            dfs_insert(d, new_task)

    # For each input option, run a dfs with that task as root to insert
    # it with all conversions. Once the dfs encounters the result of an
    # earlier inserted task with lower cost it aborts the branch.
    d = {} # { kind : (cost, task) }
    for task in options:
        dfs_insert(d, task)

    # Sort the result into a list and return it
    pre_result = [(cost, task) for kind, (cost, task) in d.iteritems()]
    pre_result.sort()
    result = [task for cost, task in pre_result]
    return result

class GreedyAdditionFinder(object):
    """
    Find the cheapest way to add a number of operands together, when
    combining all two-term addition computations and (one-term) conversions.

    We suppose that we want to find the cost for only the cheapest computation
    that can be performed and then cut off.

    Currently this is very slow, massive cutoffs and caching possible beyond this,
    though the algorithm will remain exponential in number of kinds involved at
    least.
    """
    def __init__(self, addition_cache):
        self.addition_cache = addition_cache
        self.cost_map = addition_cache.cost_map
        self.nodes_visited = 0

    def lookup_addition_cache(self, compiled_nodes):
        metas = [node.metadata for node in compiled_nodes]
        # Add the cost of the input tasks to all output costs
        base_cost = sum([node.total_cost for node in compiled_nodes])
        utils.sort_by(compiled_nodes, metas)
        metas.sort()
        d = self.addition_cache.get_computations(metas)
        # Convert the results of the task to Task TODO Refactor so
        # that Task takes arguments on execution and is not pre-bound to
        # concrete leafs, thus making it possible for the addition_cache to
        # store this directly.
        result = []
        for kind, (cost, p, adder, lconvs, rconvs) in d.iteritems():
            if p == (1, 0):
                lconvs, rconvs = rconvs, lconvs
            # apply conversions
            converted_nodes = []
            for cnode, convs in zip(compiled_nodes, [lconvs, rconvs]):
                for conv in convs:
                    conv_cost = conv.get_cost([cnode.metadata]).weigh(self.cost_map)
                    conv_metadata = cnode.metadata.copy_with_kind(conv.target_kind)
                    cnode = CompiledNode(conv, conv_cost, [cnode], conv_metadata)
                converted_nodes.append(cnode)
            # do the addition
            converted_metas = [node.metadata for node in converted_nodes]
            utils.sort_by(converted_nodes, converted_metas)
            converted_metas.sort()
            add_cost = adder.get_cost(converted_metas).weigh(self.cost_map)
            add_metadata = metadata.meta_add(converted_metas).copy_with_kind(adder.target_kind)
            add_node = CompiledNode(adder, add_cost, converted_nodes, add_metadata)
            result.append(add_node)
        if len(result) == 0:
            return None
        else:
            result.sort()
            return result[0]

    def find_cheapest_addition(self, operands):
        print 'TODO addition shuffle'
        for op in operands:
            assert isinstance(op, CompiledNode)
        self.nodes_visited += 1
        n = len(operands)
        if n == 1:
            return operands[0]
        elif n == 2:
            result = self.lookup_addition_cache(operands)
            return result
        else:
            options = []
            for left_operands, right_operands in set_of_pairwise_nonempty_splits(operands):
                left_cnode = self.find_cheapest_addition(left_operands)
                right_cnode = self.find_cheapest_addition(right_operands)
                if left_cnode is not None and right_cnode is not None:
                    added_cnode = self.lookup_addition_cache([left_cnode, right_cnode])
                    options.append(added_cnode)
                    
                #for base_cost, ltask, rtask in options:
                #    post_add_options.extend(self.lookup_addition_cache([ltask, rtask]))
                # Zip together the two lists combining the costs, then do the sort, then
                # recurse from cheapest to more expensive
                #pre_add_options = []
                #for lcost, ltask in left_options:
                #    for rcost, rtask in right_options:
                #        pre_add_options.append((lcost + rcost, ltask, rtask))
                #pre_add_options.sort()
                #post_add_options = []
                #for base_cost, ltask, rtask in options:
                #    post_add_options.extend(self.lookup_addition_cache([ltask, rtask]))
                #if len(post_add_options) > 0:
                #    post_add_options.sort()
                #    options.append(post_add_options[0])
            if len(options) > 0:
                options.sort()
                return options[0]
            else:
                return None


def generate_direct_computations(node):
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

def find_cheapest_direct_computation(node, cost_map):
    min_cost = {}
    best_task_node = {}
    for task_node in generate_direct_computations(node):
        cost = sum([task.cost for task in task_node.task_dependencies],
                   cost_value.zero_cost)
        cost_scalar = cost.weigh(cost_map)
        kind = task.metadata.kind
        if cost_scalar < min_cost.get(kind, np.inf):
            min_cost[kind] = cost_scalar
            best_task_node[kind] = (cost_scalar, task_node)
    return best_task_node

# Utilities
def multiply_if_not_single(children):
    if len(children) == 1:
        return children[0]
    else:
        return symbolic.MultiplyNode(children)

def reduce_best_tasks(tasks_lst):
    # tasks_lst is a list [ { kind : (cost, task) }, { kind : (...) ]
    # This function produces a single task-dict taking the cheapest for
    # each kind
    min_cost = {}
    result = {}
    for tasks in tasks_lst:
        for kind, (cost, task) in tasks.iteritems():
            if cost < min_cost.get(kind, np.inf):
                min_cost[kind] = cost
                result[kind] = (cost, task)
    return result
    

def cheapest_cnode(cnode_a, cnode_b):
    if cnode_b is None:
        return cnode_a
    elif cnode_a is None:
        return cnode_b
    else:
        return cnode_a if cnode_a.total_cost <= cnode_b.total_cost else cnode_b

class GreedyCompilation():
    """
    During tree traversal, each node returns
    { kind : (cost, task) } for the *optimal* task for each kind, while
    the other solutions are ignored.

    The following restrictions/heuristics are used to keep things tractable
    at the moment. Note that this can mean that even if an expression can
    be computed by the rules in the system, this compiler may not find the
    solution!

     - Currently only a single conversion will be considered, not a chain of
       them in a row (this should be fixed). TODO: Also conversions in multiplications
       are not present as of this writing.

     - When processing addition and multiplication, all the children are first
       computed and the cheapest way of computing the child picked, regardless
       of the resulting matrix kind (hence the "greedy" aspect). However within
       a single multiplication/addition node, all the options are tried.
       
     - Either the distributive rule is used on all terms, or it is not. I.e.,
       (a + b + c) * x can turn into (a * x + b * x + c * x), but
       NOT (a + b) * x + c * x.

     - When the distributive rule is used, one always computes the 'distributee'
       first and reuse it. I.e., for (a + b) * c * e, one will always get
       (a * [c * e] + ...); never ((a * c) * e + ...); where [c * e] denotes
       one specific chosen computation for [c * e] (the one that minimizes the
       total cost of the sum).

    """
    def __init__(self):
        self.cost_map = cost_value.default_cost_map
        self.cache = {}
        self.conversion_cache = ConversionCache(self.cost_map)
        self.addition_cache = AdditionCache(self.conversion_cache)
        self.addition_finder = GreedyAdditionFinder(self.addition_cache)
    
    def compile(self, root):
        self.minimum_possible_cost = 0
        self.nodes_visited = 0
        result = self.cached_visit(root)
        if result is None:
            raise ImpossibleOperationError()
        else:
            return result

    def cached_visit(self, node):
        self.nodes_visited += 1
        if node in self.cache:
            result = self.cache[node]
        else:
            result = node.accept_visitor(self, node)
            self.cache[node] = result
        if result is not None:
            assert result.arg_count == node.leaf_count
        return result

    def best_task(self, options):
        best_task = None
        best_cost = np.inf
        for cost, task in options:
            if cost < best_cost:
                best_task = task
                best_cost = cost
        return (best_cost, best_task)        

    def apply_distributive_rule(self, distributor, distributee, direction):
        # In the case of (a * b) * c -> a * c + b * c; (a * b) is 'distributor' and
        # c is 'distributee'
        if not isinstance(distributor, symbolic.AddNode):
            return None
        add_snode = distributor

        # Find out which leaves/arguments belong to distributee, for use in constructing shuffle
        start_of_distributee_args = 0 if direction == 'right' else distributor.leaf_count
        distributee_arg_indices = tuple(range(start_of_distributee_args,
                                              start_of_distributee_args + distributee.leaf_count))
        
        # Compute the distributee, and the multiply the result in with the
        # children of the add_node
        distributee_cnode = self.cached_visit(distributee)
        if distributee_cnode is None:
            return None
        distributee_sleaf = symbolic.MatrixMetadataLeaf(distributee_cnode.metadata)

        compiled_terms = []
        shuffle = []
        substitution_indices = []
        arg_start = 0 if direction == 'left' else distributee.leaf_count
        subst_index = 0 if direction == 'right' else distributor.leaf_count

        new_term_leaf_counts = [term.leaf_count + 1 for term in add_snode.children]
        if direction == 'right':
            substitution_indices = list(utils.cumsum([0] + new_term_leaf_counts[:-1]))
        else:
            substitution_indices = [i - 1 for i in utils.cumsum(new_term_leaf_counts)]

        for term in add_snode.children:
            arg_stop = arg_start + term.leaf_count
            term_arg_indices = tuple(range(arg_start, arg_stop))
            # Form symbolic tree using distributee_snode leaf 
            if direction == 'left':
                term_snode = symbolic.multiply([term, distributee_sleaf])
                shuffle.extend(term_arg_indices + distributee_arg_indices)
            elif direction == 'right':
                term_snode = symbolic.multiply([distributee_sleaf, term])
                shuffle.extend(distributee_arg_indices + term_arg_indices)
            # Compile the tree
            compiled_term = self.cached_visit(term_snode)
            if compiled_term is None:
                # Couldn't distribute
                return None
            compiled_terms.append(compiled_term)
            arg_start = arg_stop

        # Find the addition operation for adding together the compiled terms
        new_add_snode = symbolic.add([symbolic.MatrixMetadataLeaf(term_cnode.metadata)
                                      for term_cnode in compiled_terms])
        new_add_cnode = self.cached_visit(new_add_snode)
        new_add_cnode = new_add_cnode.substitute(compiled_terms)

        r = new_add_cnode.substitute(dict((i, distributee_cnode) for i in substitution_indices),
                                     flat_shuffle=shuffle)
        return r

    def visit_multiply(self, node):        
        # We parenthize expression and compile each resulting part greedily, at least
        # for now. Considering the entire multiplication expression non-greedily should
        # be rather tractable though.
        if len(node.children) > 2:
            # Break up expression using associative rule, trying each split position
            best_cnode = None
            for i in range(1, len(node.children)):
                left = multiply_if_not_single(node.children[:i])
                right = multiply_if_not_single(node.children[i:])
                cnode = self.cached_visit(multiply_if_not_single([left, right]))
                best_cnode = cheapest_cnode(best_cnode, cnode)
            return best_cnode
        else:
            left, right = node.children

            # Try to apply distributive rule
            best_cnode = self.apply_distributive_rule(left, right, 'left')
            best_cnode = cheapest_cnode(best_cnode, self.apply_distributive_rule(right, left, 'right'))

            # Compute children; ignoring any ConjugateTransposeNode's (i.e. computing their
            # children)
            def visit_with_transpose(node):
                transpose = isinstance(node, symbolic.ConjugateTransposeNode)
                if transpose:
                    node, = node.children
                cnode = self.cached_visit(node)
                return cnode, transpose
            left_cnode, left_is_transposed = visit_with_transpose(left)            
            right_cnode, right_is_transposed = visit_with_transpose(right)
            
            if left_cnode is None or right_cnode is None:
                # impossible to compute directly; return whatever came out of
                # using the distributive rule
                return best_cnode

            # When operands are transposed, we can either convert A.h -> A, or
            # try rules of the kind A.h * A. For now, be greedy in taking the
            # cheapest A.h -> A operation we can find, but try both A.h * A as
            # well as A.h->B & B * A. (TODO, fix this, this is not the place to
            # be cheap as there's no explosion).

            def find_transpose_computation(cnode):
                return self.find_best_direct_computation(
                    cnode.metadata.kind.h.get_key(), [cnode], [cnode.metadata], cnode.metadata)

            def maybe_transpose_kind(x, transpose):
                return x.h if transpose else x

            def maybe_transpose_meta(x, transpose):
                return x.transpose() if transpose else x
            
            left_options = [
                (left_cnode, left_is_transposed),
                (find_transpose_computation(left_cnode), False)]
            right_options = [
                (right_cnode, right_is_transposed),
                (find_transpose_computation(right_cnode), False)]

            for left_cnode, left_is_transposed in left_options:
                for right_cnode, right_is_transposed in right_options:
                    if left_cnode is None or right_cnode is None:
                        continue
                    
                    left_meta = maybe_transpose_meta(left_cnode.metadata, left_is_transposed)
                    right_meta = maybe_transpose_meta(right_cnode.metadata, right_is_transposed)
                    metas = [left_meta, right_meta]
                    key = (maybe_transpose_kind(left_meta.kind, left_is_transposed) *
                           maybe_transpose_kind(right_meta.kind, right_is_transposed)).get_key()
                    cnode = self.find_best_direct_computation(key, [left_cnode, right_cnode], metas,
                                                              metadata.meta_multiply(metas))
                    best_cnode = cheapest_cnode(best_cnode, cnode)

            return best_cnode


    def find_best_direct_computation(self, key, child_cnodes, metas, target_meta):
        best_cnode = None
        computations_by_kind = metas[0].kind.universe.get_computations(key)
        for target_kind, computations in computations_by_kind.iteritems():
            typed_target_meta = target_meta.copy_with_kind(target_kind)
            for computation in computations:
                cost = computation.get_cost(metas).weigh(self.cost_map)
                cnode = CompiledNode(computation, cost, child_cnodes, typed_target_meta)
                best_cnode = cheapest_cnode(best_cnode, cnode)
        return best_cnode
        
    def visit_add(self, node):
        # Recurse to compute cheapest way of computing each operand
        compiled_children = [self.cached_visit(child) for child in node.children]
        # For each operand, temporarily replace 
        
        # For addition, we do consider all possible permutations
        # (we want, e.g., CSC + Diagonal + CSR + Dense to work the right way)
        result = self.addition_finder.find_cheapest_addition(compiled_children)
        return result

    def visit_metadata_leaf(self, node):
        return CompiledNode.create_leaf(node.metadata)
            
    def visit_task_leaf(self, node):
        1/0
        return 0, node.as_task()

    def visit_decomposition(self, node):
        # TODO: For now assume that decomposition is exactly "kind.f -> kind"
        # which returns a single matrix with the exact same metadata...
        assert node.decomposition is decompositions.Factor
        compiled_child = self.cached_visit(node.child)
        kind = compiled_child.metadata.kind
        computation, = kind.universe.get_computations(kind.f.get_key())[kind]
        cost = computation.get_cost([compiled_child.metadata]).weigh(self.cost_map)
        return CompiledNode(computation, cost, [compiled_child], compiled_child.metadata)
    

def find_cost(computation, meta_args):
    assert all(isinstance(x, MatrixMetadata) for x in meta_args)
    return computation.get_cost(meta_args)


class BaseCompiler(object):
    def __init__(self):
        self.cache = {}

    def compile(self, expression):
        meta_tree, args = transforms.metadata_transform(expression)
        result = self.cache.get(meta_tree, None)
        if result is None:
            compilation = self.compilation_factory()
            result = compilation.compile(meta_tree)
            self.cache[meta_tree] = result
            if hasattr(compilation, 'stats'):
                self.stats = compilation.stats
        return result, args

class ShortestPathCompiler(BaseCompiler):
    compilation_factory = ShortestPathCompilation

class DepthFirstCompiler(BaseCompiler):
    compilation_factory = DepthFirstCompilation

class RightToLeftCompiler(BaseCompiler):
    compilation_factory = RightToLeftCompilation

class GreedyCompiler(BaseCompiler):
    compilation_factory = GreedyCompilation

    def compile(self, expression):
        meta_tree, args = transforms.metadata_transform(expression)
        _, index_args = transforms.ImplToMetadataTransform().execute(meta_tree)
        compiled_tree = self.cache.get(meta_tree, None)
        if compiled_tree is None:
            compilation = self.compilation_factory()
            self.cache[meta_tree] = compiled_tree = compilation.compile(meta_tree)
        result = compiled_tree.convert_to_task_graph([x.as_task() for x in index_args])
        result = symbolic.TaskLeaf(result, [])
        return result, args

#default_compiler_instance = ShortestPathCompiler()
#default_compiler_instance = DepthFirstCompiler()
default_compiler_instance = GreedyCompiler()

