import sys
import numpy as np

# TODO: Computers should be reentrant/thread-safe, since they can
# be assigned to a global configuration variable.

from . import formatter
from .symbolic import ExpressionNode, LeafNode, ConjugateTransposeNode
from .matrix import Matrix
from .core import ImpossibleOperationError

class DescriptionWriter(object):
    def __init__(self, stream):
        self.stream = stream
        self._indent = ''
        self._names = {}

    def format_arg(self, arg):
        if isinstance(arg, (str, int)):
            return arg
        name = self._names.get(id(arg), None)
        if name is not None:
            return name
        elif isinstance(arg, LeafNode):
            return arg.name
        elif isinstance(arg, ExpressionNode):
            name_to_matrix = {}
            s = formatter.default_formatter.format(arg, name_to_matrix)
            return s
        else:
            return arg

    def putln(self, line, *args, **kw):
        args = list(args)
        for idx, arg in enumerate(args):
            args[idx] = self.format_arg(arg)
        for key, arg in kw.iteritems():
            kw[key] = self.format_arg(arg)
        self.stream.write('%s%s\n' % (self._indent, line.format(*args, **kw)))
        
    def indent(self):
        if self._indent == '':
            self._indent = ' - '
        else:
            self._indent = '  ' + self._indent

    def dedent(self):
        self._indent = self._indent[2:]

    def register_buffer(self, name, obj):
        self._names[id(obj)] = name

class NoopWriter(object):
    def putln(self, line, *args, **kw):
        pass
    def indent(self):
        pass
    def dedent(self):
        pass
    def register_buffer(self, name, obj):
        pass

class MatVecComputer(object):
    # Not reentrant, create new instance per call to describe

    # Visitor arguments:
    #   expr - Expression node to multiply with vec
    #   vec - NumPy array
    #   out - Output buffer
    #   should_accumulate - Whether to accumulate in output (+=) or overwrite
    #
    # TODO For now, use lots of temporary buffers; this should be
    # improved.
    def __init__(self, writer, should_noop):
        self.noop = should_noop
        self.writer = writer

    def compute(self, expr, vec, out, should_accumulate):
        self.nbufs = 0
        self.refcounts = {}
        self.incref_buffer(vec)
        self.incref_buffer(vec)
        self.incref_buffer(out)
        self.incref_buffer(out)
        self.writer.register_buffer("$in", vec)
        self.writer.register_buffer("$out", out)
        expr.accept_visitor(self, expr, vec, out, should_accumulate)

    def allocate_buffer(self, dtype, shape):
        name = '$buf%d' % self.nbufs
        self.nbufs += 1
        buf = np.empty(shape, dtype)
        self.writer.register_buffer(name, buf)
        self.writer.putln("Allocate {0}: {1} buffer of shape {2}",
                          buf, dtype, shape)
        self.refcounts[id(buf)] = 1
        return buf

    def incref_buffer(self, buf):
        self.refcounts[id(buf)] = self.refcounts.setdefault(id(buf), 0) + 1

    def decref_buffer(self, buf):
        self.refcounts[id(buf)] -= 1
        if self.refcounts[id(buf)] == 0:
            # Caller must make sure to not hold references to argument!
            self.writer.putln("Deallocate {0}", buf)
            del self.refcounts[id(buf)]

    def visit_add(self, expr, vec, out, should_accumulate):
        # Multiply vec by the sum of expr.children, starting from rightmost
        self.writer.putln("Compute {0} = ({1}) * {2}:", out, expr, vec)
        self.writer.indent()
        child = expr.children[-1]
        child.accept_visitor(self, child, vec, out, False)
        for child in expr.children[-2::-1]:
            child.accept_visitor(self, child, vec, out, True)
        self.writer.dedent()

    def visit_multiply(self, expr, vec, out, should_accumulate):
        # Simply multiply from right to left. We simply
        # use a lot of temporary buffers for the time being
        self.writer.putln("Compute {0} = ({1}) * {2}:", out, expr, vec)
        self.writer.indent()
        for child in expr.children[:0:-1]:
            shape = (expr.nrows,) + vec.shape[1:]
            buf = self.allocate_buffer(expr.dtype, shape)
            child.accept_visitor(self, child, vec, buf, False)
            self.decref_buffer(vec)
            vec = buf; del buf
        child = expr.children[0]
        child.accept_visitor(self, child, vec, out, should_accumulate)
        self.decref_buffer(vec)
        self.writer.dedent()

    def visit_leaf(self, expr, vec, out, should_accumulate):
        matrix_impl = expr.matrix_impl
        if not hasattr(matrix_impl, 'apply'):
            raise NotImplementedError('Conversions on matvec not yet implemented')
        self.writer.putln("Multiply with {0} matrix: {1} {op} {2} * {3}",
                          type(expr.matrix_impl).name, out, expr, vec,
                          op='+=' if should_accumulate else '=')
        matrix_impl.apply(vec, out, should_accumulate)
            
    def visit_inverse(self, expr, vec, out, should_accumulate):
        raise NotImplementedError()

    def visit_conjugate_transpose(self, expr, vec, out, should_accumulate):
        raise NotImplementedError()


class StupidComputation(object):
    # Each visitor method returns either a LeafNode or a
    # ConjugateTransposeNode with a LeafNode child.
    
    def __init__(self, multiply_graph, writer, noop):
        self.multiply_graph = multiply_graph
        self.writer = writer
        self.noop = noop

    def is_right_vector(self, expr):
        return expr.ncols == 1 and expr.nrows > 1

    def is_left_vector(self, expr):
        return expr.nrows == 1 and expr.ncols > 1
    
    def compute(self, expr):
        return expr.accept_visitor(self, expr)

    def visit_add(self, expr):
        pass

    def visit_multiply(self, expr):
        # See StupidComputer for details on the behaviour below
        def transpose(expr):
            if isinstance(expr, ConjugateTransposeNode):
                return expr.child
            else:
                return ConjugateTransposeNode(expr)
        
        def mul_pair(left, right):
            ## # Deal with any transpositions; when dealing with vector,
            ## # we optimize this
            ## post_transpose = False
            ## if isinstance(left, ConjugateTransposeNode) and self.is_right_vector(right):
            ##     # Turn ``A.h * u`` into ``(u.h * A).h``
            ##     post_transpose = True
            ##     left, right = transpose(right), left.child
            ## elif isinstance(right, ConjugateTransposeNode) and self.is_left_vector(left):
            ##     # Turn ``u * A.h`` into ``(A * u.h).h``
            ##     post_transpose = True
            ##     left, right = right.child, transpose(left)

            # Recurse to compute left and right expressions and turn them
            # into LeafNodes.
            left = left.accept_visitor(self, left)
            right = right.accept_visitor(self, right)

            # Perform multiplications
            left_impl = left.matrix_impl
            right_impl = right.matrix_impl
            try:
                matrix_impl = self.multiply_graph.perform((left_impl, right_impl))
            except ImpossibleOperationError:
                # Try the transpose product
                left_impl = left_impl.conjugate_transpose()
                right_impl = right_impl.conjugate_transpose()
                matrix_impl = self.multiply_graph.perform((right_impl, left_impl))
                matrix_impl = matrix_impl.conjugate_transpose()
                
            result = LeafNode(None, matrix_impl)
            return result
        
        assert len(expr.children) >= 2
        # Figure out if this is a "matrix-vector" product, in which
        # case we change the order of multiplication.
        if self.is_right_vector(expr):
            # Right-to-left
            right = expr.children[-1]
            for left in expr.children[-2::-1]:
                right = mul_pair(left, right)
            return right
        else:
            # Left-to-right
            left = expr.children[0]
            for right in expr.children[1:]:
                left = mul_pair(left, right)
            return left

    def visit_leaf(self, expr):
        return expr
            
    def visit_inverse(self, expr):
        raise NotImplementedError()

    def visit_conjugate_transpose(self, expr):
        child = expr.child
        child = child.accept_visitor(self, child)
        if isinstance(child, ConjugateTransposeNode):
            return child
        else:
            return LeafNode(None, child.matrix_impl.conjugate_transpose())


class StupidComputer(object):
    """
    Computes an expression using some simple syntax-level rules.
    First, we treat all matrices with one 1-length dimension as
    a "vector". Then, ignoring any cost estimates (A and B
    are matrices, u is a vector):

      - Matrix-vector products are performed such that there's always
        a vector; ``A * B * x`` is performed right-to-left and
        ``x * A * B`` is performed left-to-right. Similarly,
        ``(A + B) * x`` is computed as ``A * x + B * x``.
        
      - Matrix-matrix products such as ``A * B * C`` are performed
        left-to-right (no matter what). Also, expressions are computed
        as formed: ``(A + B) * X`` first computes ``A + B`` before
        multiplying with ``X``.

      ##- ``A.h * u`` is computed as ``(u.h * A).h``, but ``A.h * B``
      ##  does an actual transposition of ``A``.

      - Matrix additions ``A + B`` are performed in some arbitrary
        order.

      - ``A.i * B`` always first attempts ``A.solve_right(B)``,
        then ``A.inverse() * B``.

      - ``A.h.i * B`` first tries ``A.solve_left(B)``, then
        ``A.conjugate_transpose().i * B``.

    Note that vectors and matrices are treated quite differently,
    and that the only thing qualifying a matrix as a "vector" is
    its shape.

    We always assume that the right conversions etc. are present so
    that the expression can be computed in the fashion shown above.
    The output type is not selectable, it just becomes whatever it
    is.

    No in-place operations or buffer reuse is ever performed.
    """
    
    def __init__(self, multiply_graph=None):
        if multiply_graph is None:
            from .core import multiply_graph
        self.multiply_graph = multiply_graph

    def compute(self, matrix, verbose=False, noop=False, stream=sys.stderr):
        if verbose:
            writer = DescriptionWriter(stream)
        else:
            writer = NoopWriter()
        matrix_impl = StupidComputation(self.multiply_graph, writer, noop).compute(matrix._expr)
        return Matrix(matrix_impl)


