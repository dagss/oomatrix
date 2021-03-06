from .common import *

from ..cost_value import *

def test_basic():
    yield eq_, 4 * FLOP, CostValue(FLOP=4)
    yield eq_, 4 * FLOP, 4 * FLOP
    yield eq_, 4 * FLOP, FLOP * 4
    yield ne_, 4 * FLOP, 5 * FLOP
    yield ne_, 4 * FLOP, 4 * MEM
    yield ne_, 4 * FLOP, 4

def test_identities():
    yield eq_, FLOP, FLOP + 0

def test_zero():
    yield eq_, '0', repr(FLOP - FLOP)
    yield eq_, '0', repr(0 * FLOP)
    yield eq_, '3.00e+00 MEMOP', repr(FLOP + 3 * MEMOP - FLOP)
    

def test_distributive_law():
    yield eq_, 4 * FLOP + 6 * MEM, 2 * (2 * FLOP + 3 * MEM)

def test_repr():
    yield eq_, '4.00e+00 FLOP + 2.00e+00 MEM', repr(4 * FLOP + 2 * MEM)

def test_weigh():
    yield eq_, 5, CostValue(FLOP=1, MEM=2).weigh(dict(MEM=2, FLOP=1))
    yield eq_, 5, (FLOP + 2 * MEM).weigh(dict(MEM=2, FLOP=1))
    yield eq_, 0, CostValue().weigh({})
