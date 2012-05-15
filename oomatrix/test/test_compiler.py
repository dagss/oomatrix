import re

from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl, MatrixKind
from ..computation import (computation, conversion, ImpossibleOperationError,
                           FLOP, UGLY)
from ..compiler import ExhaustiveCompiler
from .. import compiler, formatter
from ..task import LeafTask, Task

from .mock_universe import MockKind, MockMatricesUniverse

def test_outer():
    lst = list(compiler.outer([1,2,3], [4,5,6], [7,8]))
    assert lst == [(1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8),
                   (1, 6, 7), (1, 6, 8), (2, 4, 7), (2, 4, 8),
                   (2, 5, 7), (2, 5, 8), (2, 6, 7), (2, 6, 8),
                   (3, 4, 7), (3, 4, 8), (3, 5, 7), (3, 5, 8),
                   (3, 6, 7), (3, 6, 8)]

class FormatTaskExpression(formatter.BasicExpressionFormatter):
    def __init__(self):
        formatter.BasicExpressionFormatter.__init__(self, {})
        self.task_names = {}
        self.name_counter = 0

    def register_task(self, task):
        name = 'T%d' % self.name_counter
        self.name_counter += 1
        self.register_task_with_name(task, name)
        return name

    def register_task_with_name(self, task, name):
        self.task_names[task] = name

    def get_task_name(self, task):
        return self.task_names[task]

    def is_task_registered(self, task):
        return task in self.task_names
    
    def visit_leaf(self, expr):
        assert isinstance(expr, symbolic.Promise)
        return True, self.task_names[expr.task]
    
def serialize_task(lines, task, formatter):
    if isinstance(task, LeafTask):
        assert isinstance(task.descriptive_expression, symbolic.LeafNode)
        name = task.descriptive_expression.name
        formatter.register_task_with_name(task, name)
        return name
    elif not formatter.is_task_registered(task):
        # must 'compute' task
        task_name = formatter.register_task(task)
        for arg in task.argument_tasks:
            serialize_task(lines, arg, formatter)
        expr_str = formatter.format(task.descriptive_expression)
        lines.append('%s = %s' % (task_name, expr_str))
        return task_name
    else:
        return formatter.get_task_name(task)


def remove_blanks(x):
    return re.sub('\s', '', x)

def assert_compile(expected_task_graph, expected_transposed, matrix):
    compiler = ExhaustiveCompiler()
    task, is_transposed = compiler.compile(matrix._expr)
    assert expected_transposed == is_transposed
    task_lines = []
    formatter = FormatTaskExpression()
    serialize_task(task_lines, task, formatter)
    task_str = '; '.join(task_lines)
    assert remove_blanks(expected_task_graph) == remove_blanks(task_str)
    

def test_basic():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A + B, A)
    assert_compile('T0 = a + b', False, a + b)
    assert_compile('T1 = b + b; T0 = a + T1', False, a + b + b)

def test_distributive():
    ctx = MockMatricesUniverse()
    A, a, au, auh = ctx.new_matrix('A') 
    B, b, bu, buh = ctx.new_matrix('B')
    ctx.define(A * B, A)
    ctx.define(B * B, A)
    assert_compile('''
    T2 = b + b;
    T1 = a * T2;
    T3 = b * T2;
    T0 = T1 + T3
    ''', False, (a + b) * (b + b))
    

