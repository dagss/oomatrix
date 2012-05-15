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
            return task.value
        else:
            task_id = self.num_tasks
            self.num_tasks += 1
            arg_strs = []
            for arg_task, arg_result in zip(task.argument_tasks,
                                            arguments):
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



class Explainer(object):
    def __init__(self, stream, symbolic_root, task, margin=''):
        print symbolic_root
        print task.dump()
        return
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
        self.process(self.computable_root)

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

    def process(self, node):
        node.accept_visitor(self, node)

    def describe_operation(self, result, computation_name, args,
                           result_kind_name, cost):
        self.putln('{0} = {1}({2}) [{3} result, cost={4}]',
                   result, computation_name, ', '.join(args),
                   result_kind_name, cost)

    def process_operation(self, node, computation_name):
        labels = []
        expr_str = self.format_expression(node.symbolic_expr)
        self.putln('Compute {0} [cost={1}]:', expr_str, node.cost)
        self.indent()
        for child in node.children:
            labels.append(child.accept_visitor(self, child))
        label = self.get_label(node)
        self.describe_operation(label,
                                computation_name,
                                labels,
                                node.kind.name,
                                node.computation_cost,
                                )
        self.dedent()
        return label

    def visit_computable(self, node):
        return self.process_operation(node, node.computation.name)

    def visit_leaf(self, node):
        label = self.get_label(node)
        return label
        #name = node.name
        #if name is None:
        #    name = 'a %s matrix' % node.matrix_impl.name
        #return name

    def visit_conjugate_transpose(self, node):
        node.child.accept_visitor(self, node.child)

    def visit_decomposition(self, node):
        #print node, node.__dict__
        return self.process_operation(node,
                                      node.decomposition.get_name(node.kind))
    

class NoopWriter(object):
    def putln(self, line, *args, **kw):
        pass
    def indent(self):
        pass
    def dedent(self):
        pass
    def register_buffer(self, name, obj):
        pass
