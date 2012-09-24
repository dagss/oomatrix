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

class Program(object):
    """
    Mainly a list of ComputeStatement that we provide a nicer repr for.
    """
    def __init__(self, statements, matrix_names):
        self.statements = statements
        self.matrix_to_name = dict(matrix_names)
        self.name_to_matrix = dict((name, m) for m, name in matrix_names.iteritems())

    def get_matrix_name(self, matrix):
        result = self.matrix_to_name.get(matrix, None)
        if result is None:
            result = self.matrix_to_name[matrix] = self.invent_matrix_name(matrix)
            self.name_to_matrix[result] = matrix
        return result

    def invent_matrix_name(self, matrix):
        if isinstance(matrix, str):
            # Temporaries identified by string -- TBD, hacky? 
            return matrix
        elif matrix.name is not None:
            prefix = name = matrix.name
            i = 1
            while True:
                existing_matrix = self.name_to_matrix.get(name, None)
                if existing_matrix is None or matrix is existing_matrix:
                    break
                # Another matrix previously encountered shares the same name!,
                # so add suffix
                name = '%s_%d' % (prefix, i)
                i += 1
            return name
        else:
            1/0


    def __repr__(self):
        return '<oomatrix.Program:[\n  %s\n]>' % '\n  '.join(x.format(self) for x in self.statements)

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
        cleaned_args = []
        for arg in args:
            name, matrix_impl = self._parse_arg(arg)
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
            cleaned_args.append(matrix_impl)
        return cleaned_args, matrix_to_name
        
    def schedule(self, cnode, args):
        program = []
        pool = {}
        names = {}
        cleaned_args, matrix_to_name = self._parse_args(args)        
        ret_var = self._schedule(cnode, tuple(cleaned_args), program, pool)
        return Program(program, matrix_to_name)

    def _schedule(self, cnode, args, program, pool):
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
            child_result = self._schedule(child, child_args, program, pool)
            child_results.append(child_result)

        result_name = 'T%d' % len(program)
        statement = ComputeStatement(result_name, cnode.computation, child_results,
                                     cnode.weighted_cost)
        program.append(statement)
        pool[(cnode, args)] = result_name
        return result_name
        
