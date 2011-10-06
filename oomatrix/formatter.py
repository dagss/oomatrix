

class ExpressionFormatter(object):

    def format(self, expr, name_to_symbol):
        if len(expr.children) == 0:
            name_to_symbol[expr.name] = expr
            return expr.name
        else:
            child_strs = []
            for childexpr in expr.children:
                s = self.format(childexpr, name_to_symbol)
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
        return terms[0] + '.H'

    def visit_inverse(self, terms):
        assert len(terms) == 1
        return terms[0] + '.I'

default_formatter = ExpressionFormatter()

    
        
