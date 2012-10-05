from __future__ import division
import re

from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (Computation, computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY, MEMOP)
from ..compiled_node import CompiledNode
from .. import compiler, formatter, metadata, transforms, task, cost_value

from .mock_universe import (MockKind, MockMatricesUniverse, check_compilation,
                            create_mock_matrices, task_node_to_str, task_to_str, mock_meta)

import time


def mock_arg(kind):
    return task.Argument(0, mock_meta(kind))

def mock_task(kind, cost):
    comp = Computation(None, kind, kind, 'mock_%s[cost=%s]' % (kind.name, cost),
                       lambda *args: cost * FLOP)
    return task.Task(comp, cost * FLOP, [], mock_meta(kind), None)

def mock_compiled_node(kind):
    return CompiledNode.create_leaf(mock_meta(kind))

mock_cost_map = dict(FLOP=1, INVOCATION=0)

def test_compiled_node():
    # Tests basic operation, including some trivial substitutions
    ctx, (A, a), (B, b) = create_mock_matrices('A B')
    AplusB = ctx.define(A + B, B)
    AtimesB = ctx.define(A * B, A)

    # Create the tree (A * B) + B; note that B_leaf is used for both B's,
    # but this shouldn't affect anything!
    A_leaf = mock_compiled_node(A)
    B_leaf = mock_compiled_node(B)
    A_times_B_node = CompiledNode(AtimesB, 1, [A_leaf, B_leaf], mock_meta(A))
    root_a = CompiledNode(AplusB, 2, [A_times_B_node, B_leaf], mock_meta(B))
    assert root_a.leaves() == [A_leaf, B_leaf, B_leaf]
    # Now, create the same tree through substitutions
    A_plus_B_node = CompiledNode(AplusB, 2, [A_leaf, B_leaf], mock_meta(B))
    root_b = A_plus_B_node.substitute([A_times_B_node, B_leaf])
    # Test __eq__
    assert root_a == root_b and root_a is not root_b
    assert root_a == root_a.substitute(root_a.leaves())
    # Test __ne__
    assert not root_a != root_b
    assert root_a != A_leaf
    assert root_a != root_a.substitute(root_a.leaves(), shuffle=((1, 0), (2,)))
    # Make sure we don't infinitely recurse on substitution or getting leaves...
    x = A_times_B_node.substitute([A_times_B_node, B_leaf])
    assert x.children[0] is A_times_B_node
    assert x.leaves() == [A_leaf, B_leaf, B_leaf]

def test_compiled_node_substitute():
    ctx, (A, a), (B, b), (C, c) = create_mock_matrices('A B C')
    AplusA = ctx.adders[A]
    AtimesB = ctx.define(A * B, A)
    AtimesC = ctx.define(A * C, A)
    A_leaf, B_leaf, C_leaf = [mock_compiled_node(x) for x in [A, B, C]]

    # Create ((a * c) * b + (a * c) * b) as it would have occured if resulting from
    # (a * c) * (b + b); i.e. reusing (a * c)
    A_times_B_node = CompiledNode(AtimesB, 1, [A_leaf, B_leaf], mock_meta(A))
    A_times_C_node = CompiledNode(AtimesC, 1, [A_leaf, C_leaf], mock_meta(A))
    root_a = CompiledNode(AplusA, 2, [A_times_B_node, A_times_B_node], mock_meta(A),
                          shuffle=((0, 1), (0, 2)))
    root_b = root_a.substitute({0: A_times_C_node})
    A_times_C_times_B_node = CompiledNode(AtimesB, 1, [A_times_C_node, B_leaf], mock_meta(A))
    e = CompiledNode(AplusA, 2, [A_times_C_times_B_node, A_times_C_times_B_node],
                                  mock_meta(A), shuffle=((0, 1, 2), (0, 1, 3)))
    assert root_b == e


def test_compiled_node_substitute_linked():
    ctx, (A, a), (B, b), (C, c) = create_mock_matrices('A B C')
    AplusA = ctx.adders[A]
    AtimesB = ctx.define(A * B, A)
    AtimesC = ctx.define(A * C, A)
    A_leaf, B_leaf, C_leaf = [mock_compiled_node(x) for x in [A, B, C]]

    # Create ((a * c) * b + (a * c) * b) as it would have occured if resulting from
    # (a * c) * (b + b); i.e. reusing (a * c)
    A_times_B_node = CompiledNode(AtimesB, 1, [A_leaf, B_leaf], mock_meta(A))
    A_times_C_node = CompiledNode(AtimesC, 1, [A_leaf, C_leaf], mock_meta(A))
    root_a = CompiledNode(AplusA, 2, [A_times_B_node, A_times_B_node], mock_meta(A),
                          shuffle=((0, 1), (0, 2)))

    print root_a
    print
    root_b = root_a.substitute_linked((0, 2), A_times_C_node)
    print root_b
    1/0


    A_times_C_times_B_node = CompiledNode(AtimesB, 1, [A_times_C_node, B_leaf], mock_meta(A))
    e = CompiledNode(AplusA, 2, [A_times_C_times_B_node, A_times_C_times_B_node],
                                  mock_meta(A), shuffle=((0, 1, 2), (0, 1, 3)))
    assert root_b == e




def test_compiled_node_convert_to_task_graph():
    ctx, (A, a), (B, b), (C, c) = create_mock_matrices('A B C')
    AplusA = ctx.adders[A]
    AtimesB = ctx.define(A * B, A)
    BtimesC = ctx.define(B * C, B)
    # Create the tree (A * (B * C)) + (A * (B * C)), where B is *the same* input
    # argument (as would result from distributing (A + A) * (B * C)).
    A_leaf = mock_compiled_node(A)
    B_leaf = mock_compiled_node(B)
    C_leaf = mock_compiled_node(C)
    A_times_B_node = CompiledNode(AtimesB, 1, [A_leaf, B_leaf], mock_meta(A))
    B_times_C_node = CompiledNode(BtimesC, 2, [B_leaf, C_leaf], mock_meta(B))

    a = A_times_B_node.substitute([A_leaf, B_times_C_node])
    b = A_times_B_node.substitute([A_leaf, B_times_C_node])
    root = CompiledNode(AplusA, 3, [a, b], mock_meta(A), shuffle=((0, 2, 3), (1, 2, 3)))

    leaf_tasks = [mock_task(A, 1), mock_task(B, 2), mock_task(B, 3), mock_task(C, 4)]
    task = root.convert_to_task_graph(leaf_tasks)
    print task
    print
    print root
    1/0
    root_a = CompiledNode(AplusB, 2, [A_times_B_node, B_leaf], mock_meta(B))
    assert (task_to_str(task) == 'T2 = mock_A[cost=1](); T3 = mock_B[cost=2](); '
            'T1 = multiply_A_B(T2, T3); T4 = mock_B[cost=3](); T0 = add_A_B(T1, T4)')
