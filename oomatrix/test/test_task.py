from .common import *
from ..task import LeafTask, Task, Executor

META = object()

class MockTasks(object):
    def __init__(self, root_name, graph):
        self.name_graph = graph
        self.tasks = {}
        self.build(root_name)
        self.root = self.tasks[root_name]
        self.times_called = {}

    def record_call(self, task_name):
        self.times_called[task_name] = self.times_called.get(task_name, 0) + 1

    def assert_all_called_once(self):
        assert set(self.times_called.keys()) == set(self.name_graph.keys())
        assert all(x == 1 for x in self.times_called.values())

    def build(self, name):
        x = self.tasks.get(name, None)
        if x is not None:
            return x
        
        arg_names = self.name_graph.get(name, None)
        if arg_names is None:
            # leaf
            task = LeafTask(name, META)
        else:
            def compute(*args):
                self.record_call(name)
                return '%s(%s)' % (name, ','.join(args))
            arg_tasks = [self.build(arg_name) for arg_name in arg_names]
            task = Task(compute, 1, arg_tasks, META)
        self.tasks[name] = task
        return task


def test_basic():
    tasks = MockTasks('a', {'a': ['b', 'c'],
                            'b': ['d', 'e'],
                            'd': ['e']})
    e = Executor(tasks.root)
    assert 'a(b(d(e),e),c)' == e.execute()
    tasks.assert_all_called_once()
    assert len(e.refcounts) == 0
    assert len(e.results) == 0
