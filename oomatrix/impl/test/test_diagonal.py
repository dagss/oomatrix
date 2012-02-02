from ...test.common import *
from ... import Matrix

d = ndrange(4,) 
Di = Matrix(d, diagonal=True)

c = ndrange(4,) * 1j + d
Ci = Matrix(c, diagonal=True)

def test_basic():
    alleq_(d * d, (Di * Di).diagonal())
    alleq_(d + d, (Di + Di).diagonal())
    
    # square root of diagonal
    alleq_(np.sqrt(d), Di.factor().diagonal())

    # transpose multiplication
    alleq_(d * d, (Di * Di.h).diagonal())
    alleq_(c * c.conjugate(), (Ci * Ci.h).diagonal())
