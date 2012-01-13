

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

    
        
