import numpy as np

from .core import MatrixImpl
from .symbolic import (ExpressionNode, LeafNode, AddNode, MulNode, ConjugateTransposeNode,
                       InverseNode)
from .formatter import default_formatter

__all__ = ['Matrix']


class Matrix(object):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            raise TypeError("Wrong number of arguments")
        if isinstance(args[0], ExpressionNode):
            if len(args) != 1:
                raise TypeError("Wrong number of arguments")
            e = args[0]
        else:
            if len(args) != 2:
                raise TypeError("Wrong number of arguments")
            name, obj = args
            if isinstance(obj, MatrixImpl):
                r = obj
            else:
                obj = np.asarray(obj)
                diagonal = kwargs.get('diagonal', False)
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

    def _construct(self, *args, **kw):
        return type(self)(*args, **kw)

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

    def single_line_description(self, skip_name=False):
        shapestr = '%d-by-%d' % (self.ncols, self.nrows)
        typestr = '%s ' % self.get_type().name if not self.is_expression() else ''
        dtypestr = ' of %s' % self.dtype if self.dtype is not None else ''
        namestr = " '%s'" % self._expr.name if self._expr.name and not skip_name else ''
        return "%s %smatrix%s%s" % (
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
            for name, matrix in matrices.iteritems():
                lines.append('    %s: %s' %
                             (name, matrix.single_line_description(skip_name=True)))
        return '\n'.join(lines)
            

    def format_expression(self):
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
        name_to_matrix = {}
        s = default_formatter.format(self._expr, name_to_matrix)
        for key, expr in name_to_matrix.iteritems():
            name_to_matrix[key] = Matrix(key, expr.matrix_impl)
        return s, name_to_matrix

    #
    # Comparison
    #
        
    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if type(self._expr) is LeafNode and type(other._expr) is LeafNode:
            if self._expr.matrix_impl is other._expr.matrix_impl:
                # Fast path for matrices identical by id()
                return True
        raise NotImplementedError()

    def __ne__(self, other):
        return not self == other
    

    #
    # Symbolic operations. We perform sanity checks and the most
    # basic simplifications (elimination of double inverses and transposes)
    # here.
    #

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.ncols != self.ncols or self.nrows != other.nrows:
            raise ValueError('Matrices do not have same shape in addition')

        return self._construct(self._expr.symbolic_add(other._expr))

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.ncols != self.nrows:
            raise ValueError('Matrices do not conform')
        
        return self._construct(self._expr.symbolic_mul(other._expr))

    @property
    def H(self):
        if type(self._expr) is ConjugateTransposeNode:
            e = self._expr.child
        elif type(self._expr) is InverseNode:
            if type(self._expr.child) is ConjugateTransposeNode:
                e = InverseNode(self._expr.child.child)
            else:
                e = InverseNode(ConjugateTransposeNode(self._expr.child))
        else:
            e = ConjugateTransposeNode(self._expr)
        return self._construct(e)

    @property
    def I(self):
        if self.ncols != self.nrows:
            raise ValueError("Cannot take inverse of non-square matrix")
        if type(self._expr) is InverseNode:
            e = self._expr.child
        elif type(self._expr) is ConjugateTransposeNode and type(self._expr.child) is InverseNode:
            # (A**-1).H**-1 -> A.H
            e = ConjugateTransposeNode(self._expr.child.child)
        else:
            e = InverseNode(self._expr)
        return self._construct(e)

    def __pow__(self, arg):
        # Inverse
        if arg != -1:
            raise TypeError("Only -1 supported as Matrix power")
        return self.I
        
