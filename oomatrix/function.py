from __future__ import division

import hashlib
import struct
from . import utils, computation
from .cost_value import zero_cost

class Function(object):
    """
    The result of a compilation process is a tree of Function objects.

    Each Function is essentially a LISP-like program fragment;
    integers refer to input arguments while tuples are evaluated,
    e.g.;

      (f, 0, (g, 1, 2), 3, (h, (g, 1, 2), (k, 0)))

    One can typically assume that the (g, 1, 2) would be pulled out
    by a later pass; so conventionally a compiler simply emits Function
    objects in the order that is natural.

    Functions are immutable and compare by value; for comparison and
    hashing, we use a sha512 and trust that it is going to be unique.

    TODO: Figure out relationship between Computation and Function (likely
    fuse them; main difference is that Computation specifies kind while Function
    specifies metadata). For now, there's the create_from_computation constructor
    to wrap a Computation in a Function.

    The cost is the total cost of the expression and all sub-expressions
    as a scalar (i.e., weighted).
    """

    def __init__(self, expression):
        self.is_identity = False
        args_encountered = set()
        calls_encountered = set()
        self.result_metadata = expression[0].result_metadata
        self.cost, expression = self._process_expression(
            expression, args_encountered, calls_encountered)
        self.expression = expression
        self.arg_count = max(args_encountered) + 1
        if sorted(list(args_encountered)) != range(self.arg_count):
            raise ValueError('argument integer range has holes')
        
        self._make_hash()

    @staticmethod
    def create_from_computation(comp, args_metadata, result_metadata):
        # Alternate constructor, no overlap with __init__ and does not
        # call _process_expression
        self = Function.__new__(Function)
        self.is_identity = False
        self.expression = (comp,) + tuple(range(comp.arg_count))
        #self.args_metadata = tuple(args_metadata)
        self.arg_count = len(args_metadata)
        self.result_metadata = result_metadata
        self.cost = comp.get_cost(args_metadata)
        self._make_hash()
        return self

    @staticmethod
    def create_identity(metadata):
        self = Function.__new__(Function)
        self.is_identity = True
        self.result_metadata = metadata
        self.arg_count = 1
        self.expression = (0,)
        self.cost = zero_cost
        self._make_hash()
        return self

    def _process_expression(self, e, args_encountered, calls_encountered):
        # Walk through expression to a) compute cost, b) validate it.
        # Any call that is repeated (same function, same arguments) only
        # counts once towards computing cost. Of course, there's no guarantee
        # that there's not further duplicate work to eliminate if one inlined
        # the called functions... this is not dealt with currently.
        #
        # Also removes all identities
        #
        # Returns cost, transformed_tuple
        if isinstance(e, int):
            args_encountered.add(e)
            return zero_cost, e
        else:
            call, args = e[0], e[1:]
            if not isinstance(call, Function):
                raise TypeError('invalid expression')
            cost = call.cost
            # Recurse on arguments, eliminating identities along the way
            new_expr = [call]
            for arg in args:
                arg_cost, processed = self._process_expression(arg, args_encountered,
                                                               calls_encountered)
                new_expr.append(processed)
                cost += arg_cost
            if call.is_identity:
                assert len(args) == 1
                return cost, processed
            else:
                return cost, tuple(new_expr)

    def _make_hash(self):
        h = hashlib.sha512()
        h.update(self.cost.secure_hash())
        #h.update(struct.pack('Q', len(self.args_metadata)))
        h.update(self.result_metadata.secure_hash())
        #for x in self.args_metadata:
        #    h.update(x.secure_hash())
        self._hash_expression(h, self.expression)
        self._shash = h.digest()

    def _hash_expression(self, h, e):
        if isinstance(e, int):
            h.update(struct.pack('Q', e))
        else:
            h.update('(')
            call = e[0]
            if isinstance(call, Function):
                h.update(call._shash)
            elif isinstance(call, computation.Computation):
                h.update(struct.pack('Q', id(call)))
            h.update(',')
            for arg in e[1:]:
                self._hash_expression(h, arg)
                h.update(',')
            h.update(')')

    def secure_hash(self):
        return self._shash

    def __eq__(self, other):
        # Note that this definition is recursive, as the comparison of children will
        # end up doing an element-wise comparison
        if type(other) is not Function:
            return False
        else:
            return self._shash == other._shash

    def __hash__(self):
        return hash(self._shash)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return '<Function:%s:%s %s\n>' % (self.cost, self.result_metadata.kind.name,
                                          FunctionFormatter().format(self))

    def __repr__(self):
        return '<Function:%s:%s %s>' % (self.cost, self.result_metadata.kind.name,
                                        FunctionFormatter().format_expression(self))


class FunctionFormatter:
    def format(self, func):
        if func.expression == (0,):
            return '(0,)'
        else:
            self.lines = []
            self.function_names = {}
            self.visited_functions = set()
            self.process(func, is_root=True)
            return '\n'.join(self.lines)

    def format_expression(self, func):
        self.function_names = {}
        return self._format_expression(func.expression)[1]        

    def process(self, func, is_root):
        new_functions, expr_repr = self._format_expression(func.expression)
        if is_root:
            self.lines.append(expr_repr)
            self.lines.append('  where:')
        else:
            self.lines.append('    %s := %s' % (self.function_names[func], expr_repr))
        for called_func in new_functions:
            # may have been visited by earlier sibling; check
            if called_func not in self.visited_functions:
                self.visited_functions.add(called_func)
                self.process(called_func, False)                

    def _format_expression(self, e):
        if e == (0,):
            return [], '(0,)'
        new_functions = []
        call, args = e[0], e[1:]
        if isinstance(call, Function):
            name = self.function_names.get(call, None)
            if name is None:
                self.function_names[call] = name = 'f%d' % len(self.function_names)
                new_functions.append(call)
        elif isinstance(call, computation.Computation):
            name = call.name
        else:
            raise AssertionError('`call` is of type %s' % type(call))

        arg_strs = []
        for arg in args:
            if isinstance(arg, int):
                arg_strs.append('$%d' % arg)
            else:
                got_new_functions, arg_s = self._format_expression(arg)
                arg_strs.append(arg_s)
                new_functions.extend(got_new_functions)
        expression_str = '(%s %s)' % (name, ' '.join(arg_strs))
        return new_functions, expression_str



