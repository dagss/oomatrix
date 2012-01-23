"""
Superclasses for concrete matrix implementations.

The class itself (metaclass) of a MatrixImpl is a MatrixKind. E.g, if
one has::

    class Diagonal(MatrixImpl):
        ...

Then ``isinstance(Diagonal, MatrixKind)`` holds true. Matrix kinds
can be used in arithmetic operations in order to create matcher
patterns, such as::

    @computation(Diagonal * Diagonal + Dense.h, Dense)
    def foo(di1, di2, de): ..

The patterns implements comparison and can be sorted, in a way which ignores
addition commutativity, i.e., ``Dense + Diagonal == Diagonal + Dense``.
The ``get_key`` will return the same tree in both those cases.




Kind universes
--------------

New kinds are created by instantiating classes, and it would be bothersome
to have to explicitly assign them to a collection of available kinds. At
the same time, one needs to create lots of mock kinds for unit test purposes
etc., and one does not wish these to in any way modify global state.

So to avoid a global state or global registry of matrix types, we instead
have the `MatrixKindUniverse`, accessed by the `universe` attribute of each
kind. Each kind starts out with a universe consisting only of itself.
Then, each time a match pattern is created (e.g., ``Dense + Diagonal``),
one merges the implied universes.
"""

from functools import total_ordering
import threading

from .core import ConversionGraph
from .utils import argsort

#
# kind universes
#
class MatrixKindUniverse(object):
    """Keeps track of the set of all MatrixKind and Computation instances

    But note that 'all' is relative -- it means a set of kinds and
    computations which have been in touch with one another. It is
    possible to have multiple disjoint universes.

    Joining universes happens through linking them together; actual
    data about the universe is only stored in the root node. We delay
    shortening the linked lists created until the data is accessed
    through a given path, so that there is no need for backlinks.

    MatrixKindUniverse can be used from multiple threads
    concurrently. Writing (with associated locking) mostly only
    happens during program startup. Given the very special usecase, a
    single global lock seems appropriate.

    When _linked is set, the other attributes (_kinds, _computations)
    freeze and should not be consulted. However they are left around, in
    case concurrent readers are currently checking them.
    """

    # write lock for writes to *all* universes in the process
    reentrant_write_lock = threading.RLock()
    
    def __init__(self):
        self._kinds = set()
        self._computations = {}
        self._linked = None

    def _get_root(self):
        # Returns the root node, possibly shortening the chain of links
        # in the process
        linked = self._linked
        if linked is None:
            # self is the root node
            return self
        elif linked._linked is None:
            # we're linked straight at the root node, can't do better
            return linked
        else:
            # we're left dangling from a previous join; shorten the path
            with MatrixKindUniverse.reentrant_write_lock:
                # _linked NEVER gets set to None except in the ctor; so
                # we don't need the typical re-check-upon-locking
                linked = self._linked = linked._get_root()
            return linked

    def join_with(self, other):
        my_root = self._get_root()
        other_root = other._get_root()
        if my_root is other_root:
            return # already linked
        with MatrixKindUniverse.reentrant_write_lock:
            # note that it is left to future _get_root calls to update any
            # other nodes pointing to other

            # first, check again in case of race
            my_root = self._get_root()
            other_root = other._get_root()
            if my_root is other_root:
                return

            # do the join between the respective roots
            my_root._kinds.update(other_root._kinds)
            my_root._computations.update(other_root._computations)
            # NOTE: We need to update the references in the old root
            # as well -- another thread may well be READING on the
            # node at this point (i.e. be right between _get_root()
            # and accessesing the root), and need to find something. We could
            # leave the old data, but this way the dicts will
            # be deallocated.
            #
            # This means that during a race, _kinds and _computations could be
            # out of sync, but that should be OK. We could stick them in a tuple
            # to make the assigment atomic.
            other._computations = my_root._computations
            other._kinds = my_root._kinds
            
            other_root._linked = my_root
            other._linked = my_root # not strictly necesarry

    def add_kind(self, kind):
        with MatrixKindUniverse.reentrant_write_lock:
            # Need to do this atomically, so that root doesn't become non-root
            # under our nose
            root = self._get_root()
            root._kinds.add(kind)

    def add_computation(self, match_pattern, out_kind, computation):
        match_key = match_pattern.get_key()
        with MatrixKindUniverse.reentrant_write_lock:
            # Need to do this atomically, so that root doesn't become non-root
            # under our nose
            root = self._get_root()
            computation_db = root._computations

            try:
                same_match_dict = computation_db[match_key]
            except KeyError:
                computation_db[match_key] = same_match_dict = {}

            try:
                same_out_kind_lst = same_match_dict[out_kind]
            except KeyError:
                same_match_dict[out_kind] = same_out_kind_lst = []

            same_out_kind_lst.append(computation)

    def get_computations(self, key):
        """
        key: A tuple-tree representation
        """
        return self._get_root()._computations[key]

    def get_kinds(self):
        # make a copy to be safe for now
        return set(self._get_root()._kinds)

#
# Core classes
#

@total_ordering # implements __ge__ and so on
class PatternNode(object):
    def get_key(self):
        return ((self.symbol,) + 
                tuple(child.get_key() for child in self.get_sorted_children()))

    def get_sorted_children(self):
        # overriden in AddPatternNode, which is the only one which
        # is allowed to sort
        return self.children

    def __add__(self, other):
        if not isinstance(self, PatternNode):
            raise TypeError('wrong type')
        return AddPatternNode([self, other])

    def __mul__(self, other):
        if not isinstance(self, PatternNode):
            raise TypeError('wrong type')
        return MultiplyPatternNode([self, other])

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        else:
            assert not isinstance(self, MatrixKind)
            return self.children == other.children

    def __lt__(self, other):
        return self.get_key() < other.get_key()
        
    @property
    def h(self):
        return ConjugateTransposePatternNode(self)

    @property
    def i(self):
        return InversePatternNode(self)

class MatrixKind(type, PatternNode):
    _transpose_classes = {}
    
    def __init__(cls, name, bases, dct):
        type.__init__(cls, name, bases, dct)
        # Register pending conversion registrations
        to_delete = []
        pending = ConversionGraph._global_pending_conversion_registrations
        for func, decorator_args in pending.iteritems():
            if dct.get(func.__name__, None) is func:
                graph, dest_kind = decorator_args
                graph.conversion_decorator(cls, dest_kind)(func)
                to_delete.append(func)
        for func in to_delete:
            del pending[func]
        if 'name' not in dct:
            cls.name = name
        cls.universe = MatrixKindUniverse()
        cls.universe.add_kind(cls)

    def __repr__(cls):
        return "<kind:%s>" % cls.name

    def __eq__(cls, other_cls):
        return cls is other_cls

    def __lt__(cls, other_cls):
        # sort by id in this particular run, or _sort_id if available
        
        if not isinstance(other_cls, MatrixKind):
            # A single MatrixKind is always less than a more complicated
            # pattern/tuple
            return True

        # to provide stable sorting in testcases,
        # we have _sort_id
        key = getattr(cls, '_sort_id', id(cls))
        otherkey = getattr(other_cls, '_sort_id', id(other_cls))
        return key < otherkey

    def get_key(cls):
        return cls
 
    @property
    def H(cls):
        # TODO: deprecate in favor of pattern-matching/lowercase .h
        """
        A property for creating a new MatrixKind (a new class),
        representing the conjugate transpose.
        """
        result = getattr(cls, 'conjugate_transpose_class', None)
        if result is not None:
            return result
        if cls not in MatrixKind._transpose_classes:
            class NewClass(MatrixImpl):
                name = 'conjugate transpose %s' % cls.name
                def __init__(self, wrapped):
                    self.wrapped = wrapped
                    self.nrows, self.ncols = wrapped.ncols, wrapped.nrows
                def conjugate_transpose(self):
                    return self.wrapped 
                def get_element(self, i, j):
                    return self.wrapped.get_element(j, i) # TODO conj
            NewClass.__name__ = 'ConjugateTranspose%s' % cls.__name__
            MatrixKind._transpose_classes[cls] = NewClass
        return MatrixKind._transpose_classes[cls]

class MatrixImpl(object):
    __metaclass__ = MatrixKind
    
    left_shape = None
    right_shape = None
    dtype = None

    def get_type(self):
        return type(self)

    def conjugate_transpose(self):
        transpose_cls = type(self).H
        return transpose_cls(self)


def credits(library=None, authors=None):
    """
    Decorator used to decorate implementation actions with information
    about the library used and references.
    """
    attrs = {}
    if library is not None:
        attrs['library'] = library
    if authors is not None:
        attrs['authors'] = authors
    def dec(func):
        if getattr(func, 'credits', None) is None:
            func.credits = {}
        func.credits.update(attrs)
        return func
    return dec


#
# Kind expression pattern matching tree. MatrixKind instances
# are the leaf nodes.
#

class IllegalPatternError(TypeError):
    pass

class ArithmeticPatternNode(PatternNode):
    def __init__(self, children):
        # Avoid nesting arithmetic nodes of the same type;
        # "a * b * c", not "(a * b) * c".
        unpacked_children = []
        for child in children:
            if type(child) is type(self):
                unpacked_children.extend(child.children)
            else:
                unpacked_children.append(child)
        del children
        self.children = unpacked_children
        self._child_sort()
        # Join universes
        universe = self.children[0].universe
        for x in self.children[1:]:
            universe.join_with(x.universe)
        self.universe = universe

    def _child_sort(self):
        pass

class AddPatternNode(ArithmeticPatternNode):
    symbol = '+'

    def _child_sort(self):
        self.child_permutation = argsort(self.children)
        self.sorted_children = [self.children[i]
                                for i in self.child_permutation]

    def get_sorted_children(self):
        # overriden in AddPatternNode, which is the only one which
        # is allowed to sort
        return self.sorted_children

class MultiplyPatternNode(ArithmeticPatternNode):
    symbol = '*'

class SingleChildPatternNode(PatternNode):
    def __init__(self, child):
        self.child = child
        self.children = [child]
        self.universe = child.universe

class ConjugateTransposePatternNode(SingleChildPatternNode):
    symbol = 'h'
    def __init__(self, child):
        if not isinstance(child, (InversePatternNode, MatrixKind)):
            raise IllegalPatternError(
                '.h must be applied directly to A or A.i for a '
                'matrix kind A; (A + B).h etc. is not allowed')
        SingleChildPatternNode.__init__(self, child)

class InversePatternNode(SingleChildPatternNode):
    symbol = 'i'    
    def __init__(self, child):
        if not isinstance(child, MatrixKind):
            raise IllegalPatternError(
                '.i must be applied directly to a matrix kind; '
                'A.h.i should be written A.i.h, (A * B).i should '
                'be written (B.i * A.i), and so on)')
        SingleChildPatternNode.__init__(self, child)
