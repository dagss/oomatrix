

class BasicExpressionFormatter(object):
    def __init__(self, name_to_symbol):
        self.name_to_symbol = name_to_symbol
        self.anonymous_count = 0

    def format(self, expr):
        if len(expr.children) == 0:
            name = expr.name
            if name is None:
                name = '$%d' % self.anonymous_count
                self.anonymous_count += 1
            self.name_to_symbol[name] = expr
            return name
        else:
            child_strs = []
            for childexpr in expr.children:
                s = self.format(childexpr)
                if expr.precedence > childexpr.precedence:
                    s = '(%s)' % s
                child_strs.append(s)
            return expr.accept_visitor(self, child_strs)

    def visit_add(self, terms):
        return ' + '.join(terms)

    def visit_multiply(self, terms):
        return ' * '.join(terms)

    def visit_conjugate_transpose(self, terms):
        assert len(terms) == 1
        return terms[0] + '.h'

    def visit_inverse(self, terms):
        assert len(terms) == 1
        return terms[0] + '.i'

class ExpressionFormatterFactory(object):
    def format(self, expr):
        name_to_symbol = {}
        s = BasicExpressionFormatter(name_to_symbol).format(expr)
        return s, name_to_symbol

default_formatter_factory = ExpressionFormatterFactory()

    
        




#
# Operation formatter
#

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
