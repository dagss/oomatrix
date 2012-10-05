from __future__ import division

import hashlib
import struct
from . import utils, computation


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

    The cost is the total cost of the expression and all sub-expressions
    as a scalar (i.e., weighted).
    """

    def __init__(self, cost, metadata, expression):
        self.cost = cost
        self.metadata = metadata
        self.expression = expression
        if isinstance(expression, int):
            raise TypeError('invalid expression')
        args_encountered = set()
        self._validate_expression(expression, args_encountered)
        self.arg_count = max(args_encountered) + 1
        if sorted(list(args_encountered)) != range(self.arg_count):
            raise ValueError('argument integers has holes')
        self._make_hash()

    def _validate_expression(self, e, args_encountered):
        if isinstance(e, int):
            args_encountered.add(e)
        else:
            call = e[0]
            if not isinstance(call, (Function, computation.Computation)):
                raise TypeError('invalid expression')
            for arg in e[1:]:
                self._validate_expression(arg, args_encountered)

    def _make_hash(self):
        h = hashlib.sha512()
        h.update(struct.pack('d', self.cost))
        h.update(self.metadata.secure_hash())
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
