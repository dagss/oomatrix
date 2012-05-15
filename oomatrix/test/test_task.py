from .common import *
from ..task import LeafTask, Task, Scheduler, DefaultExecutor
from ..cost_value import FLOP
from ..computation import computation

from .mock_universe import MockMatricesUniverse

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
        ctx = MockMatricesUniverse()
        A, a, au, auh = ctx.new_matrix('A')

        x = self.tasks.get(name, None)
        if x is not None:
            return x
        
        arg_names = self.name_graph.get(name, None)
        if arg_names is None:
            # leaf
            task = LeafTask(name, META, None)
        else:
            @computation(A * A, A, cost=1 * FLOP)
            def compute(*args):
                self.record_call(name)
                return '%s(%s)' % (name, ','.join(args))
            arg_tasks = [self.build(arg_name) for arg_name in arg_names]
            task = Task(compute, 1 * FLOP, arg_tasks, META, None)
        self.tasks[name] = task
        return task

def make_scheduler(task):
    return Scheduler(task, DefaultExecutor())

def test_basic():
    tasks = MockTasks('a', {'a': ['b', 'c'],
                            'b': ['d', 'e'],
                            'd': ['e']})
    e = make_scheduler(tasks.root)
    assert 'a(b(d(e),e),c)' == e.execute()
    tasks.assert_all_called_once()
#    assert len(e.refcounts) == 0
#    assert len(e.results) == 0


def test_lifetime():
    # x is used by b and c; check that result is kept cached from when b
    # needs it to when c needs it
    tasks = MockTasks('a', {'a': ['b', 'c'],
                            'b': ['x'],
                            'c': ['x'],
                            'x': ['w']})
    e = make_scheduler(tasks.root)
    assert 'a(b(x(w)),c(x(w)))' == e.execute()
    tasks.assert_all_called_once()

def test_lifetimes_partial_overlap():
    tasks = MockTasks('a', {'a': ['b', 'c', 'd'],
                            'b': ['x'],
                            'c': ['x', 'q'],
                            'd': ['q'],
                            'x': ['x1'],
                            'q': ['q1']})
    e = make_scheduler(tasks.root)
    assert 'a(b(x(x1)),c(x(x1),q(q1)),d(q(q1)))' == e.execute()
    tasks.assert_all_called_once()

def test_lifetimes_trailing():
    tasks = MockTasks('a', {'a': ['b', 'c'],
                            'c': ['d']})
    e = make_scheduler(tasks.root)
    assert 'a(b,c(d))' == e.execute()
    tasks.assert_all_called_once()
    

def test_dependencies_and_costs():
    a = Task(None, 1, [], META, None)
    b = Task(None, 2, [a], META, None)
    c = Task(None, 3, [a, b], META, None)
    d = Task(None, 1, [c], META, None)
    assert d.dependencies == frozenset([a, b, c]) 
    assert 6 == c.get_total_cost()
