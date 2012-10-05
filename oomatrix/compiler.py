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
import hashlib
import struct

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
from .compiled_node import CompiledNode

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
        if isinstance(distributee, symbolic.ConjugateTransposeNode):
            print 'ugly, ughly hack #432'
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
                    cnode.metadata.kind.h.get_key(), [cnode], [cnode.metadata],
                    cnode.metadata.transpose())

            def maybe_transpose_kind(x, transpose):
                return x.h if transpose else x

            def maybe_transpose_meta(x, transpose):
                return x.transpose() if transpose else x
            
            left_options = [
                (left_cnode, left_is_transposed),
                (find_transpose_computation(left_cnode), not left_is_transposed)]
            right_options = [
                (right_cnode, right_is_transposed),
                (find_transpose_computation(right_cnode), not right_is_transposed)]

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
        if None in compiled_children:
            return None
        # For addition, we do consider all possible permutations
        # (we want, e.g., CSC + Diagonal + CSR + Dense to work the right way)
        result = self.addition_finder.find_cheapest_addition(compiled_children)
        return result

    def visit_conjugate_transpose(self, node):
        print node
        1/0

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

    def visit_inverse(self, node):
        compiled_child = self.cached_visit(node.child)
        kind = compiled_child.metadata.kind
        computation, = kind.universe.get_computations(kind.i.get_key())[kind]
        cost = computation.get_cost([compiled_child.metadata]).weigh(self.cost_map)
        return CompiledNode(computation, cost, [compiled_child], compiled_child.metadata)

def find_cost(computation, meta_args):
    assert all(isinstance(x, MatrixMetadata) for x in meta_args)
    return computation.get_cost(meta_args)


class BaseCompiler(object):
    def __init__(self):
        self.cache = {}

    def compile_as_task(self, expression):
        meta_tree, args = transforms.metadata_transform(expression)
        result = self.cache.get(meta_tree, None)
        if result is None:
            compilation = self.compilation_factory()
            result = compilation.compile(meta_tree)
            self.cache[meta_tree] = result
            if hasattr(compilation, 'stats'):
                self.stats = compilation.stats
        return result, args

class GreedyCompiler(BaseCompiler):
    compilation_factory = GreedyCompilation

    def _compile(self, expression):
        meta_tree, args = transforms.metadata_transform(expression)
        _, index_args = transforms.ImplToMetadataTransform().execute(meta_tree)
        compiled_tree = self.cache.get(meta_tree, None)
        if compiled_tree is None:
            compilation = self.compilation_factory()
            self.cache[meta_tree] = compiled_tree = compilation.compile(meta_tree)
        return compiled_tree, args, index_args

    def compile(self, expression):
        compiled_tree, args, index_args = self._compile(expression)
        return compiled_tree, args

default_compiler_instance = GreedyCompiler()

