import re
import numpy as np

from ..cost_value import FLOP
from ..kind import MatrixImpl, MatrixKind
from ..computation import computation, conversion, Computation
from .. import formatter, Matrix, symbolic, compiler, kind, metadata, scheduler
from ..task import Task, Argument
from ..symbolic import MatrixMetadataLeaf
from ..scheduler import BasicScheduler

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
        
mock_cost_map = dict(FLOP=1, INVOCATION=0)

class MockMatricesUniverse:
    def __init__(self):
        self.reset()
        self.kind_count = 0
        self.adders = {}

    def reset(self):
        self.computation_index = 0

    def define(self, match, result_kind, reprtemplate='', name=None, cost=1 * FLOP):
        if isinstance(cost, (int, float)):
            cost *= FLOP
        if name is None:
            name = match_tree_to_name(match)
        reprtemplate = '(%s)' % reprtemplate
        # If '#' is in reprtemplate, substitute it with the number of
        # times called
        times_called = [0]
        @computation(match, result_kind, cost=lambda *args: cost, name=name)
        def comp(*args):
            template = reprtemplate.replace('#', str(self.computation_index))
            result = result_kind(template % tuple(arg.value for arg in args),
                                 args[0].nrows, args[-1].ncols)
            self.computation_index += 1
            return result
        return comp

    def define_conv(self, from_kind, to_kind, cost=1 * FLOP):
        @conversion(from_kind, to_kind, cost=cost, name='%s:%s(%s)' %
                    (to_kind, to_kind, from_kind))
        def conv(a):
            return to_kind('%s(%s)' % (to_kind.name, a.value),
                           a.nrows, a.ncols)

    def new_matrix(self, name_,
                   right=(), right_h=(), add=(),
                   result='self', addition_cost=1 * FLOP):
        class NewKind(MockKind):
            name = name_
            _sort_id = name_

        if isinstance(addition_cost, int):
            addition_cost *= FLOP

        self.kind_count += 1

        if result == 'self':
            result_kind = NewKind
        else:
            result_kind = result.get_type()

        # Always have within-kind addition
        @computation(NewKind + NewKind, NewKind,
                     cost=lambda a, b: addition_cost,
                     name='add_%s_%s' % (name_, name_))
        def add(a, b):
            return NewKind('(%s + %s)' % (a.value, b.value),
                           a.nrows, b.ncols)
        self.adders[NewKind] = add
        
        return (NewKind,
                Matrix(NewKind(name_.lower(), 3, 3), name_.lower()),
                Matrix(NewKind(name_.lower() + 'u', 3, 1), name_.lower() + 'u'),
                Matrix(NewKind(name_.lower() + 'uh', 1, 3),
                       name_.lower() + 'uh'))


def create_mock_matrices(matrix_names, addition_costs=None):
    if isinstance(matrix_names, str):
        matrix_names = matrix_names.split()
    ctx = MockMatricesUniverse()
    result = (ctx,)
    if addition_costs is None:
        addition_costs = [1 * FLOP] * len(matrix_names)
    for name, cost in zip(matrix_names, addition_costs):
        A, a, au, auh = ctx.new_matrix(name, addition_cost=cost)
        result += ((A, a),)
    return result

def remove_blanks(x):
    return re.sub('\s', '', x)

class CheckScheduler(BasicScheduler):
    # For historical reasons, change variable names used in generated Programs
    # when used for checking
    def __init__(self):
        BasicScheduler.__init__(self)
        self._numvars = 1
        
    def _get_result_varname(self):
        return 'T0'

    def _get_temporary_varname(self):
        r = 'T%d' % self._numvars
        self._numvars += 1
        return r

def serialize_program(program, sep='; '):
    lines = []
    for stat in program.statements:
        lines.append('%s = %s(%s)' % (stat.result, stat.computation.name,
                                      ', '.join(stat.args)))
    return sep.join(lines)

def check_compilation(compiler_obj, expected_task_graphs, matrix):
    tree, args = compiler_obj.compile(matrix._expr)
    args = tuple(args)

    program = CheckScheduler().schedule(tree, args)
    task_str = serialize_program(program)
    if isinstance(expected_task_graphs, str):
        expected_task_graphs = [expected_task_graphs]
    expected_task_graphs = [remove_blanks(x) for x in expected_task_graphs]
    assert remove_blanks(task_str) in expected_task_graphs

def mock_meta(kind):
    return metadata.MatrixMetadata(kind, (3,), (3,), np.double)

def mock_computation(kind, cost=1):
    return Computation(None, kind, kind, 'mock_%s[cost=%s]' % (kind.name, cost),
                       lambda *args: cost * FLOP)
