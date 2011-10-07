import numpy as np

from . import formatter
from .symbolic import ExpressionNode, LeafNode

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
            
    def visit_conjugate_transpose(self, terms):
        raise NotImplementedError()

    def visit_inverse(self, terms):
        raise NotImplementedError()

