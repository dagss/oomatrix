from . import matrix, symbolic, kind
from .function import Function
from .computation import Computation


class ComputeStatement(object):
    def __init__(self, result, computation, args, cost):
        self.result = result
        self.computation = computation
        self.args = args
        self.cost = cost

    def format(self, program):
        result_name = program.get_matrix_name(self.result)
        arg_names = [program.get_matrix_name(arg) for arg in self.args]
        return '%s = %s(%s)' % (
            result_name, self.computation.name,
            ', '.join(arg_names))

    def __repr__(self):
        return '<oomatrix.ComputeStatement %s %s %s>' % (repr(self.result), self.computation,
                                                         self.args)

class DeleteStatement(object):
    def __init__(self, var):
        self.var = var

    def format(self, program):
        return 'del %s' % program.get_matrix_name(self.var)

class Program(object):
    """
    Mainly a list of ComputeStatement that we provide a nicer repr for.
    """
    def __init__(self, statements, matrix_names):
        self.statements = statements
        self.matrix_to_name = dict(matrix_names)
        self.name_to_matrix = dict((name, m) for m, name in matrix_names.iteritems())

    def get_matrix_name(self, matrix):
        if isinstance(matrix, str):
            return matrix # temporary
        else:
            # input argument
            try:
                return self.matrix_to_name[matrix]
            except IndexError:
                raise RuntimeError('matrix name unknown to Program')

    def __repr__(self):
        return '<oomatrix.Program:[\n  %s\n]>' % '\n  '.join(x.format(self) for x in self.statements)

    def execute(self):
        variables = dict(self.name_to_matrix)
        for stat in self.statements:
            if type(stat) is ComputeStatement:
                fetched_args = [variables[key] for key in stat.args]
                variables[stat.result] = stat.computation.compute(fetched_args)
            elif type(stat) is DeleteStatement:
                del variables[stat.var]
        result = variables['$result']
        return result

class BasicScheduler(object):
    def __init__(self):
        pass

    def _parse_arg(self, arg):
        if isinstance(arg, matrix.Matrix):
            if isinstance(arg._expr, symbolic.LeafNode):
                return arg._expr.name, arg._expr.matrix_impl
            else:
                raise ValueError("Cannot pass uncompiled AST as program argument to scheduler")
        elif isinstance(arg, kind.MatrixImpl):
            return None, arg
        elif isinstance(arg, symbolic.LeafNode):
            return arg.name, arg.matrix_impl
        else:
            raise TypeError("unknown argument type")

    def _parse_args(self, args):
        # Resolve matrix names so that they are unique
        matrix_to_name = {}
        name_to_matrix = {}
        arg_names = []
        unnamed_input_count = 0
        for arg in args:
            name, matrix_impl = self._parse_arg(arg)
            if matrix_impl in matrix_to_name:
                name = matrix_to_name[matrix_impl]
            else:
                if name is None:
                    name = 'input_%d' % unnamed_input_count
                    unnamed_input_count += 1
                prefix = name
                i = 1
                while True:
                    existing_matrix = name_to_matrix.get(name, None)
                    if existing_matrix is None or matrix_impl is existing_matrix:
                        break
                    # Another matrix previously encountered shares the same name!,
                    # so add suffix
                    name = '%s_%d' % (prefix, i)
                    i += 1
                matrix_to_name[matrix_impl] = name
                name_to_matrix[name] = matrix_impl
            arg_names.append(name)
        return arg_names, matrix_to_name

    def _get_temporary_varname(self):
        return 'T%d' % len(self.program)

    def _get_result_varname(self):
        return '$result'        
    
    def schedule(self, cnode, args):
        # First construct the basic program (linearize it)
        program = self._interpret(cnode, args)
        # Then delete temporary results at the point they're not needed
        program = delete_temporaries(program)
        return program
    #
    # Interpreter/linearization
    #   

    def _interpret(self, cnode, args):
        self.program = program = []
        self.pool = {}
        arg_names, matrix_to_name = self._parse_args(args)
        ret_varname = self._get_result_varname()
        ret_var = self._interpret_function(cnode, tuple(arg_names), ret_varname)
        assert ret_var == ret_varname
        del self.program
        del self.pool
        return Program(program, matrix_to_name)

    def _interpret_function(self, func, args, result_variable):
        call, arg_exprs = func.expression[0], func.expression[1:]        
        if isinstance(call, Computation):
            result = self.pool.get((func, args), None)
            if result is None:
                if result_variable is None:
                    result_variable = self._get_temporary_varname()
                call_args = tuple(args[i] for i in arg_exprs)
                statement = ComputeStatement(result_variable, call, call_args, func.cost)
                self.program.append(statement)
                self.pool[(func, args)] = result = result_variable
            return result
        else:
            return self._interpret_expression(func.expression, args, result_variable)

    def _interpret_expression(self, e, function_args, result_variable):
        if isinstance(e, int):
            return function_args[e]
        else:            
            call, arg_exprs = e[0], e[1:]        
            call_args = tuple(self._interpret_expression(arg_expr, function_args, None)
                              for arg_expr in arg_exprs)
            return self._interpret_function(call, call_args, result_variable)
    
def delete_temporaries(program):
    n = len(program.statements)
    new_stats_reverse = []
    deleted = set()
    for i in range(n - 1, -1, -1):
        for arg in program.statements[i].args[::-1]:
            if arg not in deleted:
                deleted.add(arg)
                new_stats_reverse.append(DeleteStatement(arg))
        new_stats_reverse.append(program.statements[i])
    new_stats = new_stats_reverse[::-1]
    return Program(new_stats, program.matrix_to_name)
