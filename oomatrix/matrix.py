import numpy as np
from StringIO import StringIO

from . import symbolic, decompositions
from .kind import MatrixImpl
from .symbolic import ExpressionNode, LeafNode
from .formatter import default_formatter_factory
from .computation import ImpossibleOperationError

__all__ = ['Matrix']

def resolve_result_type(a, b):
    if (a is not None and b is not None and a is not b):
        raise NotImplementedError(
            "Unable to prioritize result types %r and %r" % (a, b))
    return a if a is not None else b

class Matrix(object):

    __array_priority__ = 10000
    
    def __init__(self, obj, name=None, diagonal=False, result_type=None):
        if name is not None and not isinstance(name, (str, unicode)):
            raise TypeError('matrix name must be a string')
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
                r = dense.Strided(obj)
            e = LeafNode(name, r)

        self._expr = e
        self.ncols, self.nrows = e.ncols, e.nrows
        self.dtype = e.dtype
        self.result_type = result_type

    def _construct(self, *args, **kw):
        return type(self)(*args, **kw)

    def get_type(self):
        """
        Returns the type of this matrix, if it is computed
        """
        return self._expr.get_type()

    def get_kind(self):
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
        raise NotImplementedError()
        if compiler is None:
            from .compiler import default_compiler_instance as compiler
        task, args = compiler.compile_as_task(self._expr)
        return task, args

    def compute(self, compiler=None, name=None):
        if isinstance(self._expr, symbolic.LeafNode):
            return self

        try:
            from .scheduler import BasicScheduler
            if compiler is None:
                from .compiler import default_compiler_instance as compiler

            compiled_tree, args = compiler.compile(self._expr)
            program = BasicScheduler().schedule(compiled_tree, args)
            result = program.execute()
            result_matrix = Matrix(result, name=name)
            return result_matrix
        except ImpossibleOperationError:
            # try vector by vector to create dense matrix
            out = np.empty((self.nrows, self.ncols), self.dtype)
            v = np.zeros((self.ncols, 1))
            for i in range(self.ncols):
                v[i] = 1
                out[:, i:i + 1] = (self * v).compute().as_array()
                v[i] = 0
            return Matrix(out)

    def explain(self, compiler=None):
        from .scheduler import BasicScheduler

        if compiler is None:
            from .compiler import default_compiler_instance as compiler

        compiled_tree, args = compiler.compile(self._expr)
        program = BasicScheduler().schedule(compiled_tree, args)
        return repr(program)

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
            #lines.append(self.format_contents_brief())
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
    def __radd__(self, other):
        if other == 0:
            return self
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other = Matrix(other)
        elif other == 0:
            return self
        if not isinstance(other, Matrix):
            raise TypeError('Matrix instance needed') # TODO implement conversions

        if other.ncols != self.ncols or self.nrows != other.nrows:
            raise ValueError('Matrices do not have same shape in addition')

        return Matrix(symbolic.add([self._expr, other._expr]))

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            other = Matrix(other)
        elif other == 1:
            return self
        elif not isinstance(other, Matrix):
            # TODO implement some conversion framework for registering vector types
            raise TypeError('Type not recognized: %s' % type(other))
        if self.ncols != other.nrows:
            self_name = "'%s'" % self._expr.name if self._expr.name else '<?>'
            other_name = "'%s'" % other._expr.name if other._expr.name else '<?>'
            raise ValueError('Matrices %s and %s do not conform: ...-by-%d times %d-by-...' % (
                self_name, other_name, self.ncols, other.nrows))
        return Matrix(symbolic.multiply([self._expr, other._expr]))

    def __rmul__(self, other):
        if isinstance(other, np.ndarray):
            other = Matrix(other)
        elif other == 1:
            return self
        else:
            raise TypeError('Type not recognized: %s' % type(other))
           
        if other.ncols != self.nrows:
            raise ValueError('Matrices do not conform: ...-by-%d times %d-by-...' % (
                other.ncols, self.nrows))

        return Matrix(symbolic.multiply([other._expr, self._expr]))
        

    @property
    def h(self):
        return Matrix(symbolic.conjugate_transpose(self._expr))

    @property
    def i(self):
        if self.ncols != self.nrows:
            raise ValueError("Cannot take inverse of non-square matrix")
        return Matrix(symbolic.inverse(self._expr))

    @property
    def f(self):
        return self.factor()

    def __pow__(self, arg):
        # Inverse
        if arg != -1:
            raise TypeError("Only -1 supported as Matrix power")
        return self.I

    #
    # Conversion
    #
    def as_array(self, order=None):
        from .impl.dense import Strided
        # TODO: Remove need for this hack
        if (isinstance(self._expr, symbolic.LeafNode) and
            self._expr.kind in (Strided,)):
            return self._expr.matrix_impl.array.copy(order)
            
        computed = self.as_kind([Strided]).compute()
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

    def as_immutable(self):
        return self

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

