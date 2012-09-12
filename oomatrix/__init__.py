# Re-exports
from .matrix import Matrix, Id
from .constructors import *

#mat = Matrix

def compute(x, *args, **kw):
    return x.compute(*args, **kw)

def explain(x, *args, **kw):
    return x.explain(*args, **kw)

def compute_array(x, *args, **kw):
    return x.compute(*args, **kw).as_array()
