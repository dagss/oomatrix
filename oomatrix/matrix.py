import numpy as np

from .core import MatrixRepresentation, AddedMatrices, MultipliedMatrices

__all__ = ['Matrix']

class Matrix(object):

    def __init__(self, name, obj, diagonal=False):
        if isinstance(obj, MatrixRepresentation):
            r = obj
        else:
            obj = np.asarray(obj)
            if diagonal:
                if obj.ndim != 1:
                    raise ValueError()
                from .repr import diagonal
                r = diagonal.DiagonalMatrixRepresentation(obj)
            else:
                if obj.ndim != 2:
                    raise ValueError()

                from .repr import dense
                if obj.flags.c_contiguous:
                    r = dense.RowMajorMatrixRepresentation(obj)
                elif obj.flags.f_contiguous:
                    r = dense.ColMajorMatrixRepresentation(obj)
                else:
                    r = dense.StridedMatrixRepresentation(obj)
            
        self.name = name
        self.representation = r
        self.left_shape = r.left_shape
        self.right_shape = r.right_shape
        self.dtype = r.dtype

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.left_shape != self.left_shape or self.right_shape != other.right_shape:
            raise ValueError('Matrices do not have same shape in addition')
        
        return Matrix(self.representation.symbolic_add(other.representation))

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.left_shape != self.right_shape:
            raise ValueError('Matrices do not conform')
        
        return Matrix(self.representation.symbolic_mul(other.representation))

    def __repr__(self):
        assert len(self.left_shape) == 1
        assert len(self.right_shape) == 1
        return "%d-by-%d %s matrix '%s' of %s" % (
            self.left_shape[0],
            self.right_shape[0],
            self.representation.name,
            self.name, self.dtype)


            
