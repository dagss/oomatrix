import numpy as np

from .core import MatrixRepresentation

__all__ = ['Matrix']

class TODO:
    name = 'TODO'

class ExpressionNode(object):

    def symbolic_add(self, other):
        return AddedMatrices([self, other])

    def symbolic_mul(self, other):
        return MultipliedMatrices([self, other])

    def get_type(self):
        return TODO

    def format_expression(self, name_to_matrix=None):
        """
        Formats the expression as a string by expanding it to all the leafs.
        
        Returns
        -------

        expression : str
            A string showing the symbolic expression
        name_to_matrix : dict
            Matrices entering into the expression, mapped by the names
            used in the expression string
            
        """
        if name_to_matrix is None:
            name_to_matrix = {}
        else:
            name_to_matrix = dict(name_to_matrix)
        s = self._format_expression(name_to_matrix)
        assert s != None
        return s, name_to_matrix

    def _format_expression(self, name_to_matrix):
        exprs = [M._format_expression(name_to_matrix) for M in self.matrices]
        return self.infix_str.join(exprs)

class AddedMatrices(ExpressionNode):
    infix_str = ' + '
    
    def __init__(self, matrices):
        self.matrices = matrices
        if len(matrices) == 0:
            raise ValueError()
        self.left_shape = matrices[0].left_shape
        self.right_shape = matrices[0].right_shape

    def symbolic_add(self, other):
        if isinstance(other, AddedMatrices):
            return AddedMatrices(self.matrices + other.matrices)
        else:
            return AddedMatrices(self.matrices + [other])


class MultipliedMatrices(ExpressionNode):
    infix_str = ' * '
    def __init__(self, matrices):
        self.matrices = matrices
        if len(matrices) == 0:
            raise ValueError()
        self.left_shape = matrices[0].left_shape
        self.right_shape = matrices[-1].right_shape

    def symbolic_mul(self, other):
        if isinstance(other, MultipliedMatrices):
            return MultipliedMatrices(self.matrices + other.matrices)
        else:
            return MultipliedMatrices(self.matrices + [other])

class Matrix(ExpressionNode):
    def __init__(self, name, obj, diagonal=False):
        if isinstance(obj, ExpressionNode):
            self._expression = obj
            self.name = name
            self._representation = None
            self.dtype = None
            self.left_shape = obj.left_shape
            self.right_shape = obj.right_shape
            return
        
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
        self._expression = None
        self._representation = r
        self.left_shape = r.left_shape
        self.right_shape = r.right_shape
        self.dtype = r.dtype

    def get_type(self):
        """
        Returns the type of this matrix, if it is computed
        """
        if self._representation is not None:
            return self._representation.get_type()
        else:
            assert self._expression is not None
            return self._expression.get_type()

    def is_expression(self):
        return self._expression is not None

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.left_shape != self.left_shape or self.right_shape != other.right_shape:
            raise ValueError('Matrices do not have same shape in addition')

        return Matrix(None, self.symbolic_add(other))

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.left_shape != self.right_shape:
            raise ValueError('Matrices do not conform')
        
        return Matrix(None, self.symbolic_mul(other.representation))

    def single_line_description(self, skip_name=False):
        shapestr = '%d-by-%d' % (self.left_shape[0],
                                 self.right_shape[0])
        typestr = self.get_type().name
        dtypestr = ' of %s' % self.dtype if self.dtype is not None else ''
        namestr = " '%s'" % self.name if self.name is not None and not skip_name else ''
        return "%s %s matrix%s%s" % (
            shapestr,
            typestr,
            namestr,
            dtypestr)

    def __repr__(self):
        assert len(self.left_shape) == 1
        assert len(self.right_shape) == 1

        if not self.is_expression():
            return self.single_line_description()
        else:
            lines = []
            lines.append('%s given by:' % self.single_line_description())
            lines.append('')
            expression, matrices = self.format_expression()
            lines.append('    ' + expression)
            lines.append('')
            lines.append('where')
            lines.append('')
            for name, matrix in matrices.iteritems():
#                assert not matrix.is_expression()
                lines.append('    %s: %s' %
                             (name, matrix.single_line_description(skip_name=True)))
            return '\n'.join(lines)
            
    def _format_expression(self, name_to_matrix):
        if self._expression is not None:
            return self._expression._format_expression(name_to_matrix)
        else:
            name = self.name
            if name is None:
                raise NotImplementedError("Leaf-matrix without name")
            try:
                x = name_to_matrix[name]
            except KeyError:
                name_to_matrix[name] = self
            else:
                if self is not x:
                    raise NotImplementedError("Two matrices with same name")
            return name
            
