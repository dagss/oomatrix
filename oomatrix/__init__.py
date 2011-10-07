# Re-exports
from .matrix import Matrix
from .vector import Vector

mat = Matrix
vec = Vector

def compute(x, *args, **kw):
    return x.compute(*args, **kw)

def explain(x, *args, **kw):
    return x.explain(*args, **kw)
