from . import matrix, symbolic, kind

class ComputeStatement(object):
    def __init__(self, result, computation, args, cost):
        self.result = result
        self.computation = computation
        self.args = args
        self.cost = cost

    def format(self, program):
        result_name = program.get_matrix_name(self.result)
        arg_names = [program.get_matrix_name(arg) for arg in self.args]
        return '%s = %s(%s) # cost=%s' % (
            result_name, self.computation.name,
            ', '.join(arg_names), self.cost)

    def __repr__(self):
        return '<oomatrix.ComputeStatement %s %s %s>' % (repr(self.result), self.computation,
                                                         self.args)

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
            fetched_args = [variables[key] for key in stat.args]
            variables[stat.result] = stat.computation.compute(fetched_args)
            #print stat.result, stat.computation.name, fetched_args
        #print '==========='
        #print
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
        
    def schedule(self, cnode, args):
        program = []
        pool = {}
        names = {}
        arg_names, matrix_to_name = self._parse_args(args)        
        ret_var = self._schedule(cnode, tuple(arg_names), program, pool, '$result')
        assert ret_var == '$result'
        return Program(program, matrix_to_name)

    def _schedule(self, cnode, args, program, pool, result_variable):
        result = pool.get((cnode, args), None)
        if result is not None:
            return result

        if cnode.is_leaf:
            return args[0]

        # First make sure all the children have been scheduled and get the
        # variable/matrix_impl their result will be stored in
        child_results = []
        for child, shuffle in zip(cnode.children, cnode.shuffle):
            child_args = tuple(args[i] for i in shuffle)
            child_result = self._schedule(child, child_args, program, pool, None)
            child_results.append(child_result)

        result_variable = 'T%d' % len(program) if result_variable is None else result_variable
        statement = ComputeStatement(result_variable, cnode.computation, child_results,
                                     cnode.weighted_cost)
        program.append(statement)
        pool[(cnode, args)] = result_variable
        return result_variable
        
