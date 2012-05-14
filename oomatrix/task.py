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
        self.refcounts = {}
        self.results = {}
        self.gray_tasks = set()

    def execute(self):
        self._incref(self.root_task)
        result = self._execute(self.root_task)
        self._decref(self.root_task)
        assert len(self.results) == 0
        return result

    def _incref(self, task):
        refs = self.refcounts.get(task, 0)
        refs += 1
        self.refcounts[task] = refs

    def _decref(self, task):
        refs = self.refcounts[task]
        refs -= 1
        if refs == 0:
            del self.results[task]
            del self.refcounts[task]
        else:
            self.refcounts[task] = refs

    def _execute(self, task):
        x = self.results.get(task, None)
        if x is not None:
            if task in self.gray_tasks:
                raise AssertionError("Cycle in graph")
            # Already computed
            return x
        
        self.gray_tasks.add(task)
        # If we depend on a and b, and a also depend on b, we want to avoid
        # multiple evaluation of b. So first, incref the result of all arguments
        for arg_task in task.argument_tasks:
            self._incref(arg_task)
        # Then recurse; if the computation "runs ahead" to following arguments
        # then the refcount causes the result to be kept in cache
        arg_values = [self._execute(arg_task)
                      for arg_task in task.argument_tasks]        
        # Perform the task
        result = task.compute(*arg_values)
        self.results[task] = result
        # Decref all intermediaries
        for arg_task in task.argument_tasks:
            self._decref(arg_task)
        # Finally, un-cache all results that are no longer needed
        self.gray_tasks.remove(task)
        return result
