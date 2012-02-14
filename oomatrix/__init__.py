# Re-exports
from .matrix import Matrix, Id

mat = Matrix

def compute(x, *args, **kw):
    return x.compute(*args, **kw)

def explain(x, *args, **kw):
    return x.explain(*args, **kw)
