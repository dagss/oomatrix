"""
Symbolic tree...

Hashing/equality: Leaf nodes compares by object identity, the rest compares
by contents.


By using the factory functions (add, multiply, conjugate_transpose,
inverse, etc.), one ensures that the tree is in a canonical shape,
meaning it has a certain number of "obvious" expression simplifications which
only requires looking at the symbolic structure. Some code expects
trees to be in this canonical state.  (TODO: Move all such
simplification from constructors to factories)

 - No nested multiplies or adds; A * B * C, not (A * B) * C
 - No nested transposes or inverses; A.h.h and A.i.i is simplified
 - Never A.h.i, but instead A.i.h
 - (A * B).h -> B.h * A.h, (A + B).h -> A.h + B.h. This makes the
   "no nested multiplies or adds rules" stronger

"""

from functools import total_ordering

from .cost_value import FLOP, INVOCATION
from .utils import argsort, invert_permutation
from . import kind, cost_value, metadata

import numpy as np

# Factory functions
def add(children):
    return AddNode(children)

def multiply(children):
    return MultiplyNode(children)

def conjugate_transpose(expr):
    if isinstance(expr, ConjugateTransposeNode):
        # a.h.h -> a
        return expr.child
    elif (isinstance(expr, InverseNode) and 
          isinstance(expr.child, ConjugateTransposeNode)):
        # a.h.i.h -> a.i
        return InverseNode(expr.child.child)
    elif isinstance(expr, MultiplyNode):
        transposed_children = [conjugate_transpose(x)
                               for x in expr.children]
        return multiply(transposed_children[::-1])
    elif isinstance(expr, AddNode):
        transposed_children = [conjugate_transpose(x)
                               for x in expr.children]
        return add(transposed_children)
    else:
        return ConjugateTransposeNode(expr)

def inverse(x):
    return InverseNode(x)

class TODO:
    name = 'TODO'

class PatternMismatchError(ValueError):
    pass

@total_ordering
class ExpressionNode(object):
    name = None
    kind = None
    _hash = None
    
    def get_type(self):
        return TODO

    def can_distribute(self):
        """
        Can one use the distributive law on this node? Overriden
        by AddNode and ConjugateTransposeNode
        """
        return False

    def distribute_right(self, other):
        raise NotImplementedError()

    def distribute_left(self, other):
        raise NotImplementedError()

    def dump(self):
        from .formatter import BasicExpressionFormatter
        return BasicExpressionFormatter({}).format(self)

    def metadata_tree(self):
        """Returns a tree with metadata leaf objects; useful for caching.
        """
        print type(self)
        raise NotImplementedError()

    def as_tuple(self):
        """Returns the tuple-serialization of the tree
        """
        return ((self.symbol,) + 
                tuple(child.as_tuple() for child in self.get_sorted_children()))

    def get_sorted_children(self):
        return self.children

    def as_computable_list(self, pattern):
        """Converts tree to an argument list matching `pattern` (a
        tree from kind.py)
        """
        raise NotImplementedError('please override')

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.symbol,
                               tuple(hash(child)
                                     for child in self.get_sorted_children())))
        return self._hash

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.as_tuple() == other.as_tuple()
        
        #if len(self.children) != len(other.children):
        #    return False
        #my_children = self.get_sorted_children()
        #other_children = other.get_sorted_children()
        #for a, b in zip(my_children, other_children):
        #    if not a == b:
        #        return False
        # Given the same arithmetic going on, and the same leaf nodes (by id),
        # then propagated properties like ncols, nrows, dtype, cost
        # and so on should also be the same.
        # We need to override in all BaseComputable nodes though.
        #return True

    def __lt__(self, other):
        return self.as_tuple() < other.as_tuple()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '\n'.join(self._repr(indent=''))

    def _attr_repr(self):
        return ''

    def _repr(self, indent):
        lines = [indent + '<%s: %s' % (type(self).__name__, self._attr_repr())]
        for child in self.children:
            lines.extend(child._repr(indent + '   '))
        lines[-1] += '>'
        return lines
      

class ArithmeticNode(ExpressionNode):
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
        # Following is correct both for multiplication and addition...
        self.nrows= self.children[0].nrows
        self.ncols = self.children[-1].ncols
        self.dtype = self.children[0].dtype # TODO combine better
        self.universe = self.children[0].universe
        #self.cost = sum(child.cost for child in self.children)
        self._child_sort()

    def metadata_tree(self):
        """Returns a tree with metadata leaf objects; useful for caching.
        """
        return type(self)([child.metadata_tree() for child in self.children])

    def _child_sort(self):
        pass

class AddNode(ArithmeticNode):
    symbol = '+'
    def _child_sort(self):
        child_keys = [child.as_tuple() for child in self.children]
        self.child_permutation = argsort(child_keys)
        self.sorted_children = [self.children[i]
                                for i in self.child_permutation]

    def get_sorted_children(self):
        return self.sorted_children
    
    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_add(*args, **kw)

    def can_distribute(self):
        return True
    
    def distribute_right(self, other):
        return AddNode([MultiplyNode([term, other])
                        for term in self.children])

    def distribute_left(self, other):
        return AddNode([MultiplyNode([other, term])
                        for term in self.children])

    def as_computable_list(self, pattern):
        if not isinstance(pattern, kind.AddPatternNode):
            raise PatternMismatchError()
        if len(self.sorted_children) != len(pattern.sorted_children):
            raise PatternMismatchError()
        
        # Now, we need to apply the inverse permutation in `pattern`
        # to figure out how the function wants to be called and how our
        # own arguments match
        sorted_to_pattern = invert_permutation(pattern.child_permutation)
        arguments_in_sorted_order = [child.as_computable_list(pattern_child)
                                     for child, pattern_child in zip(
                                         self.sorted_children,
                                         pattern.sorted_children)]
        result = []
        for i in range(len(sorted_to_pattern)):
            result.extend(arguments_in_sorted_order[sorted_to_pattern[i]])
        return result

class MultiplyNode(ArithmeticNode):
    symbol = '*'
    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_multiply(*args, **kw)

    def as_computable_list(self, pattern):
        if not isinstance(pattern, kind.MultiplyPatternNode):
            raise PatternMismatchError()
        if len(self.children) != len(pattern.children):
            raise PatternMismatchError()
        result = []
        for expr_child, pattern_child in zip(self.children, pattern.children):
            result.extend(expr_child.as_computable_list(pattern_child))
        return result

class SingleChildNode(ExpressionNode):

    def metadata_tree(self):
        return type(self)(self.child.metadata_tree())


class ConjugateTransposeNode(SingleChildNode):
    """
    Note that
    ``ConjugateTransposeNode(ConjugateTransposeNode(x)) is x``,
    so that one does NOT always have
    ``type(ConjugateTransposeNode(x)) is ConjugateTransposeNode``.
    """

    symbol = 'h'
    
    def __new__(cls, expr):
        if isinstance(expr, ConjugateTransposeNode):
            # a.h.h -> a
            return expr.child
        elif (isinstance(expr, InverseNode) and 
              isinstance(expr.child, ConjugateTransposeNode)):
                # a.h.i.h -> a.i
                return InverseNode(expr.child.child)
        else:
            # a.i.h is OK
            return ExpressionNode.__new__(cls, expr)
    
    def __init__(self, child):
        assert not isinstance(child, ConjugateTransposeNode)
        self.child = child
        self.children = [child]
        self.ncols, self.nrows = child.nrows, child.ncols
        self.universe = child.universe
        self.dtype = child.dtype
        self.kind = child.kind

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_conjugate_transpose(*args, **kw)

    def can_distribute(self):
        # Since (A + B).h = A.h + B.h:
        return self.child.can_distribute()

    def distribute_right(self, other):
        if not isinstance(self.child, AddNode):
            raise AssertionError()
        terms = self.child.children
        return AddNode(
            [MultiplyNode([term.conjugate_transpose(), other])
             for term in terms])

    def distribute_left(self, other):
        if not isinstance(self.child, AddNode):
            raise AssertionError()
        terms = self.child.children
        return AddNode(
            [MultiplyNode([other, term.conjugate_transpose()])
             for term in terms])

    def conjugate_transpose(self):
        return self.child

    def as_computable_list(self, pattern):
        if not isinstance(pattern, kind.ConjugateTransposePatternNode):
            raise PatternMismatchError()
        return self.child.as_computable_list(pattern.child)

    def compute(self):
        return ConjugateTransposeNode(LeafNode(None, self.child.compute()))

class InverseNode(SingleChildNode):
    symbol = 'i'
    
    def __new__(cls, expr):
#        print type(child), child.__dict__
        if isinstance(expr, InverseNode):
            # a.i.i -> a.i
            return expr.child
        elif isinstance(expr, ConjugateTransposeNode):
            if isinstance(expr.child, InverseNode):
                # a.i.h.i -> a.h
                return ConjugateTransposeNode(expr.child.child)
            else:
                # a.h.i -> a.i.h
                self = ExpressionNode.__new__(cls, expr.child)
                self.__init__(expr.child)
                return ConjugateTransposeNode(self)
        else:
            return ExpressionNode.__new__(cls, expr)
    
    def __init__(self, child):
        self.child = child
        self.children = [child]
        self.nrows, self.ncols = child.nrows, child.ncols
        self.universe = child.universe
        self.dtype = child.dtype
        #self.cost = child.cost

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_inverse(*args, **kw)

    def as_computable_list(self, pattern):
        if not isinstance(pattern, kind.InversePatternNode):
            raise PatternMismatchError()
        return self.child.as_computable_list(pattern.child)

class BracketNode(ExpressionNode):
    symbol = 'b'
    
    def __init__(self, child, allowed_kinds=None):
        self.child = child
        self.children = [child]
        self.allowed_kinds = allowed_kinds
        self.nrows, self.ncols = child.nrows, child.ncols
        self.universe = child.universe
        self.dtype = child.dtype
        #self.cost = child.cost

    def metadata_tree(self):
        return BracketNode(self.child.metadata_tree(), self.allowed_kinds)

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_bracket(*args, **kw)

    def call_func(self, func, pattern):
        raise NotImplementedError()

class BlockedNode(ExpressionNode):
    def __init__(self, blocks, nrows, ncols):
        # blocks should be a list of (node, selector, selector)
        nrows = 0
        ncols = 0
        for node, row_selector, col_selector in blocks:
            pass # just assert shape
        first_node = blocks[0][0]
        self.blocks = blocks
        self.nrows = nrows
        self.ncols = ncols
        self.universe = first_node.universe
        self.dtype = first_node.dtype
        self.cost = None
        

class BaseComputable(ExpressionNode):
    # TODO: REMOVE
    children = ()

    def as_computable_list(self, pattern):
        if pattern != self.kind:
            raise PatternMismatchError()
        return [self]

class Promise(BaseComputable):
    def __init__(self, task):
        metadata = task.metadata
        self.metadata = task.metadata
        self.nrows = metadata.nrows
        self.ncols = metadata.ncols
        self.kind = metadata.kind
        self.universe = self.kind.universe
        self.dtype = metadata.dtype
        self.task = task

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_leaf(*args, **kw)

    def __hash__(self):
        return id(self)

    def as_computable_list(self, pattern):
        if pattern != self.metadata.kind:
            raise PatternMismatchError()
        return [self.task]

class LeafNode(BaseComputable):
    cost = 0 * FLOP
    
    def __init__(self, name, matrix_impl):
        from .kind import MatrixImpl
        if not isinstance(matrix_impl,MatrixImpl):
            raise TypeError('not isinstance(matrix_impl, MatrixImpl)')
        self.name = name
        self.matrix_impl = matrix_impl
        self.kind = type(matrix_impl)
        self.universe = self.kind.universe
        self.nrows = matrix_impl.nrows
        self.ncols = matrix_impl.ncols
        self.dtype = matrix_impl.dtype

    def metadata_tree(self):
        return metadata.MatrixMetadata(self.kind, (self.nrows,), (self.ncols,),
                                       self.dtype)

    def compute(self):
        return self.matrix_impl

    def get_type(self):
        return type(self.matrix_impl)

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_leaf(*args, **kw)

    def as_tuple(self):
        return self

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        return id(self) < id(other)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def _attr_repr(self):
        return self.name

class DecompositionNode(ExpressionNode):
    """
    Represents a promise to perform a matrix decomposition.
    """
    def __init__(self, child, decomposition):
        self.symbol = 'decomposition:%s' % decomposition.name
        self.child = child
        self.children = [child]
        self.decomposition = decomposition
        self.nrows, self.ncols = child.nrows, child.ncols
        self.universe = child.universe
        self.dtype = child.dtype
        self.kind = child.kind

    def metadata_tree(self):
        return DecompositionNode(self.child.metadata_tree(), self.decomposition)

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_decomposition(*args, **kw)

    def as_computable_list(self, pattern):
        if not isinstance(pattern, kind.FactorPatternNode):
            raise PatternMismatchError()
        return self.child.as_computable_list(pattern.child)


for x, val in [
    (BaseComputable, 1000),
    (DecompositionNode, 1000),
    (BracketNode, 1000),
    (InverseNode, 40),
    (ConjugateTransposeNode, 40),
    (MultiplyNode, 30),
    (AddNode, 20)]:
    x.precedence = val



#
# Symbolic tree manipulation tools
#

def apply_right_distributive_rule(expr):
    """
    Applies the right-distributive rule on a MultiplyNode.

    E.g., the right distributive rule yields::

      (A + B) * x => A * x + B * x
      (A + B).h * x => A.h * x + B.h * x
      A * (C + D) => A * (C + D) # not left-distributive

      A * (B + C) * (D + E) * x =>
        A * (B * (D * x + E * x) + C * (D * x + E * x)) 
    
    
    """
    if not isinstance(expr, MultiplyNode):
        raise TypeError('expr must be MultiplyNode')
    head, tail = expr.children[0], expr.children[1:]
    # Recurse to distribute on tail
    if len(tail) == 1:
        processed_tail = tail[0]
    else:
        processed_tail = apply_right_distributive_rule(
            MultiplyNode(tail))
    if head.can_distribute():
        return head.distribute_right(processed_tail)
    else:
        return MultiplyNode([head, processed_tail])

def apply_left_distributive_rule(expr):
    """
    Applies the right-distributive rule on a MultiplyNode.
    """
    if not isinstance(expr, MultiplyNode):
        raise TypeError('expr must be MultiplyNode')
    tail, head = expr.children[:-1], expr.children[-1]
    # Recurse to distribute on tail
    if len(tail) == 1:
        processed_tail = tail[0]
    else:
        processed_tail = apply_left_distributive_rule(
            MultiplyNode(tail))
    if head.can_distribute():
        return head.distribute_left(processed_tail)
    else:
        return MultiplyNode([processed_tail, head])
