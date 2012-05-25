import numpy as np

from . import symbolic
from .cost_value import zero_cost

class Argument(object):
    """
    Placeholder for an argument; essentially represents "leaves" in the
    Task DAG.
    """
    def __init__(self, argument_index, metadata):
        self.argument_index = argument_index
        self.metadata = metadata
        self.args = ()
        self.dependencies = frozenset()

    def get_total_cost(self):
        return zero_cost

class Task(object):
    """
    *Note*: It is not the case that the total_cost of the sum of two tasks
    equals the sum of the total_cost; there could be overlapping dependencies
    """
    def __init__(self, computation, cost, args,
                 metadata, descriptive_expression):
        self.computation = computation
        self.cost = cost
        self.metadata = metadata
        self.args = args
        self.descriptive_expression = descriptive_expression
        # Compute set of all dependencies (that are Task instances)
        task_args = [arg for arg in args if isinstance(arg, Task)]
        self.dependencies = (frozenset()
                             .union(*[arg.dependencies for arg in task_args])
                             .union(task_args))
        # Compute total_cost
        self.total_cost = sum(task.cost for task in self.dependencies) + cost

    def get_total_cost(self):
        return self.total_cost

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '<Task %x:%r; total cost:%r>' % (id(self),
                                                self.computation.name if
                                                self.computation is not None
                                                else '', self.total_cost)

    def compute(self, *args):
        result = self.computation.compute(*args)
        from kind import MatrixImpl
        if isinstance(result, MatrixImpl):
            assert result.nrows == self.metadata.rows_shape[0]
            assert result.ncols == self.metadata.cols_shape[0]
        return result


    def dump_lines(self, encountered, indent=''):
        my_id = encountered.get(self, None)
        if my_id is None:
            # Dump full representation
            my_id = len(encountered)
            encountered[self] = str(my_id)
            lines = ['%s%s: %r' % (indent, my_id, self)]
            for dep in self.args:
                lines += dep.dump_lines(encountered, indent + '    ')
        else:
            # Reference earlier dumped
            lines = ['%s%s: (see above)' % (indent, my_id)]
        return lines

    def dump(self):
        return '\n'.join(self.dump_lines({}))

class DefaultExecutor(object):
    def __init__(self, expression_args):
        self.results = {}
        self.expression_args = expression_args
    
    def execute_task(self, task, arguments):
        if isinstance(task, Task):
            result = task.compute(*arguments)
        else:
            result = self.expression_args[task.argument_index]
        self.results[task] = result
        return result

    def get_result(self, task):
        return self.results[task]

    def is_ready(self, task):
        return task in self.results
    
    def release_result(self, task):
        if task in self.results:
            del self.results[task]

    def done(self):
        assert len(self.results) == 0

class Scheduler(object):

    def __init__(self, root_task, executor):
        self.root_task = root_task
        self.gray_tasks = set()
        self.executor = executor

    def execute(self):
        # First incref the entire tree; then we decref during computation
        # traversal
        result = self._execute(self.root_task, frozenset())
        self.executor.release_result(self.root_task)
        self.executor.done()
        return result

    def _execute(self, task, keep_for_parent):
        # Use recursion working from the last argument towards the first
        # in order to build a list of what we want to cache
        
        def eval_args(results, i, keep_set):
            arg_task = task.args[i]
            if i > 0:
                # Add our own deps to keep_set, and evaluate preceding args
                please_keep = keep_set.union(arg_task.dependencies)
                kept_for_me = arg_task.dependencies.difference(keep_set)
                eval_args(results, i - 1, please_keep)
            else:
                kept_for_me = ()
            # Do evaluation of this argument
            result = self._execute(arg_task, keep_set)
            results[i] = result
            if arg_task not in keep_set:
                self.executor.release_result(arg_task)
            for x in kept_for_me:
                self.executor.release_result(x)

        if self.executor.is_ready(task):
            if task in self.gray_tasks:
                raise AssertionError("Cycle in graph")
            return self.executor.get_result(task)
        
        self.gray_tasks.add(task)

        n = len(task.args)
        arg_results = [None] * n
        if n > 0:
            eval_args(arg_results, n - 1, keep_for_parent)
        result = self.executor.execute_task(task, arg_results)
        self.gray_tasks.remove(task)
        return result
