from .kind import MatrixImpl
from . import cost_value, symbolic, decompositions

atomic_nodes = (symbolic.LeafNode, symbolic.DecompositionNode)

class BasicExpressionFormatter(object):
    def __init__(self, name_to_symbol):
        self.name_to_symbol = name_to_symbol
        self.anonymous_count = 0

    def format(self, expr):
        is_atom, r = expr.accept_visitor(self, expr)
        return r

    def precedence(self, expr):
        return expr.precedence

    def format_children(self, parent_precedence, children):
        terms = []
        for child in children:
            is_atom, term = child.accept_visitor(self, child)
            do_parens = child.precedence < parent_precedence
            if do_parens:
                term = '(%s)' % term
            terms.append(term)
        return terms

    def visit_leaf(self, expr):
        name = expr.name
        if name is None:
            name = '$%d' % self.anonymous_count
            self.anonymous_count += 1
        self.name_to_symbol[name] = expr
        return True, name

    def visit_add(self, expr):
        terms = self.format_children(expr.precedence, expr.children)
        return False, ' + '.join(terms)

    def visit_multiply(self, expr):
        terms = self.format_children(expr.precedence, expr.children)
        return False, ' * '.join(terms)

    def visit_conjugate_transpose(self, expr):
        is_atom, term = expr.child.accept_visitor(self,
                                                  expr.child)
        if not is_atom:
            term = '(%s)' % term
        return True, term + '.h'

    def visit_inverse(self, expr):
        terms = self.format_children(expr.precedence, expr.children)
        assert len(terms) == 1
        return True, terms[0] + '.i'

    def visit_bracket(self, expr):
        terms = self.format_children(expr.precedence, expr.children)
        assert terms
        return True, '[%s]' % terms[0]

    def visit_decomposition(self, expr):
        terms = self.format_children(expr.precedence, expr.children)
        assert len(terms) == 1
        if expr.decomposition is decompositions.Factor:
            return True, '%s.f' % terms[0]
        else:
            return True, '%s.%s()' % (terms[0], expr.decomposition.name)

class ExpressionFormatterFactory(object):
    def format(self, expr):
        name_to_symbol = {}
        s = BasicExpressionFormatter(name_to_symbol).format(expr)
        return s, name_to_symbol

default_formatter_factory = ExpressionFormatterFactory()

    

#
# explain()
#

class ExplainingExecutor(object):
    def __init__(self, stream, expression_formatter, margin=''):
        self.stream = stream
        self.margin = margin
        self.num_indents = 0
        self.num_tasks = 0
        self.results = {}
        self.expression_formatter = expression_formatter

    # Implement Executor interface
    
    def execute_task(self, task, arguments):
        if task.computation is None:
            # leaf task
            return task.argument_index
        else:
            task_id = self.num_tasks
            self.num_tasks += 1
            arg_strs = []
            for arg_task, arg_result in zip(task.args, arguments):
                expr = arg_task.descriptive_expression
                #print expr
                #arg_str = self.expression_formatter.format(expr)
                #if arg_result in self.results:
                #    arg_str += ' (computed in task %s)' % arg_result
                if isinstance(expr, symbolic.LeafNode):
                    arg_str = self.expression_formatter.format(expr)
                else:
                    arg_str = repr(arg_result)
                arg_strs.append(arg_str)
            
            self.describe_operation(task_id, task.computation.name,
                                    arg_strs,
                                    task.metadata.kind,
                                    task.cost)
            self.results[task] = task_id
            return task_id

    def get_result(self, task):
        return self.results[task]

    def is_ready(self, task):
        return task in self.results
    
    def release_result(self, task):
        if task.computation is None:
            # leaf task
            return
        elif task in self.results:
            #self.putln('Release memory of task %d' % self.results[task])
            del self.results[task]

    def done(self):
        pass


    # Utilities
    def putln(self, line, *args, **kw):
        self.stream.write(self.margin + '    ' * self.num_indents +
                          line.format(*args, **kw) + '\n')
    
    def describe_operation(self, result, computation_name, args,
                           result_kind_name, cost):
        self.putln('{0} = {1}({2}) [{3} result, cost={4}]',
                   result, computation_name, ', '.join(args),
                   result_kind_name, cost)

    

class NoopWriter(object):
    def putln(self, line, *args, **kw):
        pass
    def indent(self):
        pass
    def dedent(self):
        pass
    def register_buffer(self, name, obj):
        pass
