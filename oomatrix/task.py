import numpy as np

from .cost_value import zero_cost

class Task(object):
    """
    *Note*: It is not the case that the total_cost of the sum of two tasks
    equals the sum of the total_cost; there could be overlapping dependencies
    """
    def __init__(self, computation, cost, argument_tasks,
                 metadata):
        self.computation = computation
        self.cost = cost
        self.metadata = metadata
        self.argument_tasks = argument_tasks
        # Compute set of all dependencies
        self.dependencies = (frozenset()
                             .union(*[arg.dependencies
                                      for arg in argument_tasks])
                             .union(argument_tasks))
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
        return self.computation.compute(*args)


    def dump_lines(self, encountered, indent=''):
        my_id = encountered.get(self, None)
        if my_id is None:
            # Dump full representation
            my_id = len(encountered)
            encountered[self] = str(my_id)
            lines = ['%s%s: %r' % (indent, my_id, self)]
            for dep in self.argument_tasks:
                lines += dep.dump_lines(encountered, indent + '    ')
        else:
            # Reference earlier dumped
            lines = ['%s%s: (see above)' % (indent, my_id)]
        return lines

    def dump(self):
        return '\n'.join(self.dump_lines({}))

class LeafTask(Task):
    def __init__(self, value, metadata):
        Task.__init__(self, None, zero_cost, [], metadata)
        self.value = value

    def compute(self, *args):
        assert len(args) == 0
        return self.value

class Executor(object):

    def __init__(self, root_task):
        self.root_task = root_task
        self.results = {}
        self.gray_tasks = set()

    def execute(self):
        # First incref the entire tree; then we decref during computation
        # traversal
        result = self._execute(self.root_task, frozenset())
        assert len(self.results) == 0
        return result

    def _execute(self, task, keep_for_parent):
        x = self.results.get(task, None)
        if x is not None:
            if task in self.gray_tasks:
                raise AssertionError("Cycle in graph")
            # Already computed
            return x
        
        self.gray_tasks.add(task)

        def eval_args(results, i, keep_set):
            arg_task = task.argument_tasks[i]
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
            if arg_task in keep_set:
                self.results[arg_task] = result
            for x in kept_for_me:
                if x in self.results:
                    del self.results[x]

        n = len(task.argument_tasks)
        arg_results = [None] * n
        if n > 0:
            eval_args(arg_results, n - 1, keep_for_parent)
        
        # Recurse; if the computation "runs ahead" to following arguments
        # then the refcount causes the result to be kept in cache
        #arg_values = [self._execute(arg_task)
        #              for arg_task in task.argument_tasks]
        # Perform the task
        result = task.compute(*arg_results)
        #self.results[task] = result

        kept_for_me = keep_for_parent

        
        # Decref all intermediaries
        #for arg_task in task.argument_tasks:
        #    self._decref_tree(arg_task)
        self.gray_tasks.remove(task)
        return result
