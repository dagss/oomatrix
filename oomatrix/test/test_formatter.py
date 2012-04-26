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
def computable_add(*x):
    return computable(add(*x))
def computable_mul(*x):
    return computable(mul(*x))

def sexp(*args):
    func = args[0]
    evaluated_args = []
    for arg in args[1:]:
        if isinstance(arg, tuple):
            arg = sexp(*arg)
        evaluated_args.append(arg)
    return func(*evaluated_args)

def test_conjuate_transpose_formatting():
    f = formatter.BasicExpressionFormatter({})
    tree = sexp(h, (add, (h, leaf('x')), (h, leaf('y'))))
    assert '(x.h + y.h).h' == f.format(tree)
    tree = sexp(h, (computable_add, (h, leaf('x')), (h, leaf('y'))))
    print f.format(tree)
    


#    <ConjugateTransposeNode: 
#   <ComputableNode: (np.conjugate and) np.dot; 7.36e+07 FLOP + 7.36e+07 MEMOP
#      <LeafNode: rhs>
#      <ComputableNode: oomatrix.impl.dense.to_column_major; 7.36e+07 MEMOP
#         <ComputableNode: oomatrix.impl.diagonal.diagonal_to_dense; 7.36e+07 MEMOP
#            <ComputableNode: oomatrix.impl.diagonal.conjugate_transpose; 0
#               <LeafNode: D>>>>>>

