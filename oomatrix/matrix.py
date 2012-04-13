import numpy as np
from StringIO import StringIO

from . import symbolic, decompositions
from .kind import MatrixImpl
from .symbolic import ExpressionNode, LeafNode
from .formatter import default_formatter_factory, Explainer

__all__ = ['Matrix']


class Matrix(object):

    __array_priority__ = 10000
    
    def __init__(self, obj, name=None, diagonal=False):
        if isinstance(obj, ExpressionNode):
            if (name, diagonal) != (None, False):
                raise TypeError("cannot provide options when passing an ExpressionNode")
            e = obj
        elif isinstance(obj, MatrixImpl):
            e = LeafNode(name, obj)
        elif isinstance(obj, (str, bool, int, tuple)):
            # Simply protect against common misuses
            raise TypeError('first argument should contain matrix data')
        elif (isinstance(obj, list) and len(obj) > 0 and
              isinstance(obj[1], (Matrix, tuple))):
            pass
        else:
            obj = np.asarray(obj)
            if diagonal:
                if obj.ndim != 1:
                    raise ValueError()
                from .impl import diagonal
                r = diagonal.Diagonal(obj)
            else:
                if obj.ndim != 2:
                    raise ValueError("array ndim != 2")
                from .impl import dense
                if obj.flags.c_contiguous:
                    r = dense.RowMajor(obj)
                elif obj.flags.f_contiguous:
                    r = dense.ColumnMajor(obj)
                else:
                    r = dense.Strided(obj)
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
        shapestr = '%d-by-%d' % (self.nrows, self.ncols)
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
        m, n = self.nrows, self.ncols
        lines = []
        for i in range(m):
            elems = [str(self[i, j]) for j in range(n)]
            s = ' '.join(elems)
            lines.append('[%s]' % s)
        return '\n'.join(lines)

    def compile(self, compiler=None): 
        if compiler is None:
            from .compiler import ExhaustiveCompiler
            compiler = ExhaustiveCompiler()
        computable = compiler.compile(self._expr)
        return computable

    def compute(self, compiler=None):
        computable = self.compile(compiler=compiler)
        return Matrix(computable.compute())

    def explain(self, compiler=None):
        computable = self.compile(compiler=compiler)
        stream = StringIO()
        Explainer(stream, self._expr, computable, margin='    ').explain()
        return stream.getvalue()

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
        s, name_to_matrix = default_formatter_factory.format(self._expr)
        for key, expr in name_to_matrix.iteritems():
            name_to_matrix[key] = Matrix(expr.matrix_impl, key)
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

        return Matrix(symbolic.AddNode([self._expr, other._expr]))

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            other = Matrix(other)
        elif not isinstance(other, Matrix):
            # TODO implement some conversion framework for registering vector types
            raise TypeError('Type not recognized')
        if self.ncols != other.nrows:
            raise ValueError('Matrices do not conform: ...-by-%d times %d-by-...' % (
                self.ncols, other.nrows))
        return Matrix(symbolic.MultiplyNode([self._expr, other._expr]))

    def __rmul__(self, other):
        raise NotImplementedError()
        # TODO: Made tricky by not wanting to conjugate a complex result
        
        #if isinstance(other, np.ndarray):
        #    return Vector(self.H._expr, other, transpose=True)
        #else:
        #   raise TypeError('Type not recognized')
        

    @property
    def h(self):
        return Matrix(symbolic.ConjugateTransposeNode(self._expr))

    @property
    def i(self):
        if self.ncols != self.nrows:
            raise ValueError("Cannot take inverse of non-square matrix")
        return Matrix(symbolic.InverseNode(self._expr))

    def __pow__(self, arg):
        # Inverse
        if arg != -1:
            raise TypeError("Only -1 supported as Matrix power")
        return self.I

    #
    # Conversion
    #
    def as_array(self, order=None):
        from .impl.dense import RowMajor, ColumnMajor, Strided
        computed = self.as_kind([RowMajor, ColumnMajor, Strided]).compute()
        if isinstance(computed._expr, symbolic.ConjugateTransposeNode):
            array = computed._expr.child.matrix_impl.array.T.conjugate()
        else:
            array = computed._expr.matrix_impl.array.copy(order)
            # todo: do not make copy if we own result
        return array

    def as_kind(self, kinds):
        if isinstance(kinds, tuple):
            raise TypeError('kinds argument should be a kind or a list')
        if not isinstance(kinds, list):
            kinds = [kinds]
        return Matrix(symbolic.BracketNode(self._expr, allowed_kinds=kinds))

    def bracket(self):
        return Matrix(symbolic.BracketNode(self._expr, allowed_kinds=None))

    #
    # array conversion
    #
    def diagonal(self):
        """Return the diagonal as an array

        If `self` is an expression, this may trigger computation of the
        entire matrix.
        """
        matrix = self.compute()
        expr = matrix._expr
        should_conjugate = False
        if isinstance(expr, symbolic.ConjugateTransposeNode):
            should_conjugate = True
            expr = expr.child
        if not isinstance(expr, LeafNode):
            raise NotImplementedError()
        diagonal = expr.matrix_impl.diagonal()
        if should_conjugate:
            diagonal = diagonal.conjugate()
        return diagonal

    #
    # decompositions
    #
    factor = decompositions.make_matrix_method(decompositions.Factor)
        

#
# Constructors
#

def Id(n):
    from .impl.scaled_identity import ScaledIdentity
    return Matrix(ScaledIdentity(1, n))

