from .kind import MatrixImpl
from .symbolic import LeafNode
from . import cost_value

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
            return expr.accept_visitor(self, expr, child_strs)

    def visit_add(self, expr, terms):
        return ' + '.join(terms)

    def visit_multiply(self, expr, terms):
        return ' * '.join(terms)

    def visit_conjugate_transpose(self, expr, terms):
        assert len(terms) == 1
        return terms[0] + '.h'

    def visit_inverse(self, expr, terms):
        assert len(terms) == 1
        return terms[0] + '.i'

    def visit_bracket(self, expr, terms):
        assert terms
        return '[%s]' % terms[0]

    def visit_decomposition(self, expr, terms):
        assert len(terms) == 1
        return '%s.%s()' % (terms[0], expr.decomposition.name)

class ExpressionFormatterFactory(object):
    def format(self, expr):
        name_to_symbol = {}
        s = BasicExpressionFormatter(name_to_symbol).format(expr)
        return s, name_to_symbol

default_formatter_factory = ExpressionFormatterFactory()

    

#
# explain()
#

class Explainer(object):
    def __init__(self, stream, symbolic_root, computable_root, margin=''):
        self.stream = stream
        self.symbolic_root = symbolic_root
        self.computable_root = computable_root
        self.overview_name_to_expr = {}
        self.expression_formatter = BasicExpressionFormatter(
            self.overview_name_to_expr)
        self.num_indents = 0
        self.margin = margin
        self.computable_names = {}
        self.temp_name_counter = 0

    def explain(self):
        expr_str = self.format_expression(self.symbolic_root)
        for name, expr in self.overview_name_to_expr.iteritems():
            self.computable_names[expr] = name
        self.stream.write('Computing expression:\n\n    %s\n\n' % expr_str)
        self.stream.write('by:\n\n')
        self.process(self.computable_root, 'Root')

    def format_expression(self, expr):
        return self.expression_formatter.format(expr)

    def get_label(self, computable):
        label = self.computable_names.get(computable, None)
        if label is None:
            label = '$tmp%d' % self.temp_name_counter
            self.temp_name_counter += 1
            #print 'creating label', label, computable
            self.computable_names[computable] = label
        return label

    def putln(self, line, *args, **kw):
        self.stream.write(self.margin + '    ' * self.num_indents +
                          line.format(*args, **kw) + '\n')

    def indent(self):
        self.num_indents += 1

    def dedent(self):
        self.num_indents -= 1

    def process(self, node, prefix):
        node.accept_visitor(self, node, prefix)

    def visit_computable(self, node, prefix):
        labels = []
        for child in node.children:
            labels.append(child.accept_visitor(self, child, ''))
        label = self.get_label(node)
        self.putln('{0} = {1}({2}) [{3} result, cost={4}]',
                   label,
                   node.computation.name,
                   ', '.join(labels),
                   node.kind.name,
                   node.cost,
                   )
        return label
        
        #self.dedent()    
        #self.putln()
        #print 'Computable', node, node.__dict__

    def visit_leaf(self, node, prefix):
        label = self.get_label(node)
        return label
        #name = node.name
        #if name is None:
        #    name = 'a %s matrix' % node.matrix_impl.name
        #return name

    def visit_conjugate_transpose(self, node, prefix):
        node.child.accept_visitor(self, node.child, prefix)

class NoopWriter(object):
    def putln(self, line, *args, **kw):
        pass
    def indent(self):
        pass
    def dedent(self):
        pass
    def register_buffer(self, name, obj):
        pass
