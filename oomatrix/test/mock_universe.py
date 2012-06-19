import re
import numpy as np

from ..cost_value import FLOP
from ..kind import MatrixImpl, MatrixKind
from ..computation import computation, conversion
from .. import formatter, Matrix, symbolic, compiler, kind
from ..task import Task, Argument
from ..symbolic import MatrixMetadataLeaf

class MockKind(MatrixImpl):
    def __init__(self, value, nrows, ncols):
        self.value = value
        self.nrows = nrows
        self.ncols = ncols
        self.dtype = np.double
        
    def __repr__(self):
        return '%s:%s' % (type(self).name, self.value)

def match_tree_to_name(node, parens_used=False):
    if isinstance(node, kind.AddPatternNode) and not parens_used:
        return 'add_' + '_'.join(['%s' % match_tree_to_name(child, True)
                                  for child in node.children])
    elif isinstance(node, kind.MultiplyPatternNode) and not parens_used:
        return 'multiply_' + '_'.join(['%s' % match_tree_to_name(child, True)
                                       for child in node.children])
    elif isinstance(node, kind.MatrixKind):
        return node.name
    elif isinstance(node, kind.InversePatternNode):
        return match_tree_to_name(node.children[0]) + 'i'
    elif isinstance(node, kind.ConjugateTransposePatternNode):
        return match_tree_to_name(node.children[0]) + 'h'
    elif isinstance(node, kind.FactorPatternNode):
        return match_tree_to_name(node.children[0]) + 'f'
    else:
        raise AssertionError('Please provide a computation name manually')
        


class MockMatricesUniverse:
    def __init__(self):
        self.reset()
        self.kind_count = 0

    def reset(self):
        self.computation_index = 0

    def define(self, match, result_kind, name=None, reprtemplate='', cost=1 * FLOP):
        if name is None:
            name = match_tree_to_name(match)                
        reprtemplate = '(%s)' % reprtemplate
        # If '#' is in reprtemplate, substitute it with the number of
        # times called
        times_called = [0]
        @computation(match, result_kind, cost=cost, name=name)
        def comp(*args):
            template = reprtemplate.replace('#', str(self.computation_index))
            result = result_kind(template % tuple(arg.value for arg in args),
                                 args[0].nrows, args[-1].ncols)
            self.computation_index += 1
            return result

    def define_conv(self, from_kind, to_kind, cost=1 * FLOP):
        @conversion(from_kind, to_kind, cost=cost, name='%s:%s(%s)' %
                    (to_kind, to_kind, from_kind))
        def conv(a):
            return to_kind('%s(%s)' % (to_kind.name, a.value),
                           a.nrows, a.ncols)

    def new_matrix(self, name_,
                   right=(), right_h=(), add=(),
                   result='self'):
        class NewKind(MockKind):
            name = name_
            _sort_id = name_

        self.kind_count += 1

        if result == 'self':
            result_kind = NewKind
        else:
            result_kind = result.get_type()

        # Always have within-kind addition
        @computation(NewKind + NewKind, NewKind,
                     cost=lambda a, b: 1 * FLOP,
                     name='add_%s_%s' % (name_, name_))
        def add(a, b):
            return NewKind('(%s + %s)' % (a.value, b.value),
                           a.nrows, b.ncols)
        
        return (NewKind,
                Matrix(NewKind(name_.lower(), 3, 3), name_.lower()),
                Matrix(NewKind(name_.lower() + 'u', 3, 1), name_.lower() + 'u'),
                Matrix(NewKind(name_.lower() + 'uh', 1, 3),
                       name_.lower() + 'uh'))



def remove_blanks(x):
    return re.sub('\s', '', x)

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
    
    def visit_metadata_leaf(self, expr):
        return True, self.task_names[expr]

    def visit_task_leaf(self, expr):
        return True, self.task_names[expr.task]
    
def serialize_task(lines, task, args, formatter):
    if isinstance(task, Argument):
        leaf_matrix = args[task.argument_index]
        name = leaf_matrix.name
        formatter.register_task_with_name(task, name)
        return name
    elif not formatter.is_task_registered(task):
        # must 'compute' task
        task_name = formatter.register_task(task)
        arg_names = [serialize_task(lines, arg, args, formatter)
                     for arg in task.args]
        expr_str = '%s(%s)' % (task.computation.name, ', '.join(arg_names))
        #expr_str = formatter.format(task.descriptive_expression)
        lines.append('%s = %s' % (task_name, expr_str))
        return task_name
    else:
        return formatter.get_task_name(task)

def check_compilation(compiler_obj, expected_task_graph, matrix):
    tree, args = compiler_obj.compile(matrix._expr)
    # todo: transpose
    assert isinstance(tree, symbolic.TaskLeaf)
    task = tree.task
    #assert expected_transposed == is_transposed
    task_lines = []
    formatter = FormatTaskExpression()
    serialize_task(task_lines, task, args, formatter)
    task_str = '; '.join(task_lines)
    assert remove_blanks(expected_task_graph) == remove_blanks(task_str)
