from .matrix import Matrix
from .vector import Vector

def compute(x, *args, **kw):
    return x.compute(*args, **kw)

def describe(x, *args, **kw):
    return x.describe(*args, **kw)
