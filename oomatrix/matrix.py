import numpy as np

from .core import MatrixImpl
from .symbolic import ExpressionNode, LeafNode, AddNode, MulNode

__all__ = ['Matrix']


class Matrix(object):
    def __init__(self, name, obj, diagonal=False):
        if isinstance(obj, ExpressionNode):
            e = obj
        else:
            if isinstance(obj, MatrixImpl):
                r = obj
            else:
                obj = np.asarray(obj)
                if diagonal:
                    if obj.ndim != 1:
                        raise ValueError()
                    from .impl import diagonal
                    r = diagonal.DiagonalImpl(obj)
                else:
                    if obj.ndim != 2:
                        raise ValueError()

                    from .impl import dense
                    if obj.flags.c_contiguous:
                        r = dense.RowMajorImpl(obj)
                    elif obj.flags.f_contiguous:
                        r = dense.ColMajorImpl(obj)
                    else:
                        r = dense.StridedImpl(obj)
            e = LeafNode(name, r)

        self._expr = e
        self.ncols, self.nrows = e.ncols, e.nrows
        self.dtype = e.dtype

    def get_type(self):
        """
        Returns the type of this matrix, if it is computed
        """
        return self._expr.get_type()

    def get_impl(self):
        if self.is_expression():
            raise ValueError("Matrix not computed")
        return self._expr.matrix_impl

    def is_expression(self):
        return type(self._expr) is not LeafNode

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.ncols != self.ncols or self.nrows != other.nrows:
            raise ValueError('Matrices do not have same shape in addition')

        return Matrix(None, self._expr.symbolic_add(other._expr))

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.ncols != self.nrows:
            raise ValueError('Matrices do not conform')
        
        return Matrix(None, self._expr.symbolic_mul(other._expr))

    def single_line_description(self, skip_name=False):
        shapestr = '%d-by-%d' % (self.ncols, self.nrows)
        typestr = self.get_type().name
        dtypestr = ' of %s' % self.dtype if self.dtype is not None else ''
        namestr = " '%s'" % self._expr.name if self._expr.name and not skip_name else ''
        return "%s %s matrix%s%s" % (
            shapestr,
            typestr,
            namestr,
            dtypestr)

    def format_contents_brief(self):
        """
        Give a representation of the contents
        """
        # TODO: Actually make this brief. This is the
        # reason for the ackward way of fetching elements:
        # Normally only the corners are fetched
        m, n = self.ncols, self.nrows
        lines = []
        for i in range(m):
            elems = [str(self[i, j]) for j in range(n)]
            s = ' '.join(elems)
            lines.append('[%s]' % s)
        return '\n'.join(lines)

    def compute(self):
        if type(self._expr) is LeafNode:
            return self
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        if self.is_expression():
            raise ValueError("Matrix not computed")
        else:
            i, j = index # todo: support slices etc.
            return self._expr.matrix_impl.get_element(i, j)

    def __repr__(self):
        lines = []
        if not self.is_expression():
            lines.append(self.single_line_description())
            lines.append(self.format_contents_brief())
        else:
            lines.append('%s given by:' % self.single_line_description())
            lines.append('')
            expression, matrices = self.format_expression()
            lines.append('    ' + expression)
            lines.append('')
            lines.append('where')
            lines.append('')
            for name, matrix_impl in matrices.iteritems():
#                assert not matrix.is_expression()
                matrix = Matrix(name, matrix_impl)
                lines.append('    %s: %s' %
                             (name, matrix.single_line_description(skip_name=True)))
        return '\n'.join(lines)
            

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
        s = self._expr.format_expression(name_to_matrix)
        return s, name_to_matrix


            
