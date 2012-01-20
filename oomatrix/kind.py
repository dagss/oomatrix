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



#There is
#however a difference in how arguments are mapped, so there's a seperate
#utility ``call_with_tree`` to call a callable with a symbolic tree.




"""

from functools import total_ordering

from .core import ConversionGraph

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


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
        if isinstance(other, MatrixKind):
            return not other < self
        elif self.symbol == other.symbol:
            return self.children < other.children
        else:
            return self.symbol < other.symbol
        
    @property
    def h(self):
        return ConjugateTransposePatternNode(self)

    @property
    def i(self):
        return InversePatternNode(self)

class MatrixKind(type, PatternNode):
    _transpose_classes = {}
    
    def __init__(cls, name, bases, dct):
        super(MatrixKind, cls).__init__(name, bases, dct)
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

    def __repr__(cls):
        return "<kind:%s>" % cls.name

    def __eq__(cls, other_cls):
        return cls is other_cls

    def __lt__(cls, other_cls):
        # sort by id in this particular run, or _sort_id if available
        
        if not isinstance(other_cls, MatrixKind):
            if not isinstance(other_cls, PatternNode):
                raise TypeError('can only compare with PatternNode')
            else:
                # A single MatrixKind is always less than a more complicated
                # pattern
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
        self._init_sorting()

    def _init_sorting(self):
        pass

class AddPatternNode(ArithmeticPatternNode):
    symbol = '+'

    def _init_sorting(self):
        self.child_permutation = argsort(self.children)
        self.sorted_children = [self.children[i]
                                for i in self.child_permutation]

    def get_sorted_children(self):
        # overriden in AddPatternNode, which is the only one which
        # is allowed to sort
        return self.sorted_children

class MultiplyPatternNode(ArithmeticPatternNode):
    symbol = '*'

class ConjugateTransposePatternNode(PatternNode):
    symbol = 'h'
    def __init__(self, child):
        if not isinstance(child, (InversePatternNode, MatrixKind)):
            raise IllegalPatternError(
                '.h must be applied directly to A or A.i for a '
                'matrix kind A; (A + B).h etc. is not allowed')
        self.child = child
        self.children = [child]

class InversePatternNode(PatternNode):
    symbol = 'i'    
    def __init__(self, child):
        if not isinstance(child, MatrixKind):
            raise IllegalPatternError(
                '.i must be applied directly to a matrix kind; '
                'A.h.i should be written A.i.h, (A * B).i should '
                'be written (B.i * A.i), and so on)')
        self.child = child
        self.children = [child]

