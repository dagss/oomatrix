from .common import *
from .. import Matrix, compute, explain, symbolic

from ..kind import MatrixImpl
from .. import symbolic as sym, formatter
from ..cost_value import CostValue

class MockImpl(MatrixImpl):
    nrows = ncols = 3
    dtype = None
impl = MockImpl()

def cost(a):
    print a
    return CostValue()

class MockComputation:
    target_kind = MockImpl
    cost = cost
    name = 'foo'
    
    #universe = MockImpl.universe

def leaf(x):
    return sym.LeafNode(x, impl)

def add(*x): return sym.AddNode(x)
def mul(*x): return sym.MultiplyNode(x)
def h(x): return sym.ConjugateTransposeNode(x)
def computable(expr):
    return sym.ComputableNode(MockComputation(), (), 3, 3, None, expr)

def sexp(*args):
    func = args[0]
    evaluated_args = []
    for arg in args[1:]:
        if isinstance(arg, str):
            arg = leaf(arg)
        elif isinstance(arg, tuple):
            arg = sexp(*arg)
        evaluated_args.append(arg)
    return func(*evaluated_args)

def test_conjuate_transpose_formatting():
    f = formatter.BasicExpressionFormatter({})
    tree = sexp(h, (add, (h, 'x'), (h, 'y')))
    yield eq_, '(x.h + y.h).h', f.format(tree)
