from .utils import argsort, invert_permutation
from . import kind

class TODO:
    name = 'TODO'

class PatternMismatchError(ValueError):
    pass

class ExpressionNode(object):
    name = None

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

    def get_key(self):
        """Returns the tuple-serialization of the tree, useful as a key

        This is in the format given in MatrixKind.get_key; i.e., only MatrixKind
        information is present, not instance information.
        """
        return ((self.symbol,) + 
                tuple(child.get_key() for child in self.get_sorted_children()))

    def get_sorted_children(self):
        return self.children

    def as_argument_list(self, pattern):
        """Converts tree to an argument list matching `pattern` (a
        tree from kind.py)
        """
        raise NotImplementedError('please override')



class LeafNode(ExpressionNode):
    children = ()
    
    def __init__(self, name, matrix_impl):
        from .kind import MatrixImpl
        if not isinstance(matrix_impl, MatrixImpl):
            raise TypeError('not isinstance(matrix_impl, MatrixImpl)')
        self.name = name
        self.matrix_impl = matrix_impl
        self.nrows, self.ncols = matrix_impl.nrows, matrix_impl.ncols
        self.dtype = matrix_impl.dtype

    def get_type(self):
        return type(self.matrix_impl)

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_leaf(*args, **kw)

    def get_key(self):
        return type(self.matrix_impl)

    def as_argument_list(self, pattern):
        if pattern != type(self.matrix_impl):
            raise PatternMismatchError()
        return [self.matrix_impl]

class ComputableNode(ExpressionNode):
    def __init__(self, kind, nrows, ncols, dtype, computable):
        self.kind = kind
        self.nrows = nrows
        self.ncols = ncols
        self.dtype = dtype
        self.computable = computable

    def as_argument_list(self, pattern):
        if pattern != self.kind:
            raise PatternMismatchError()
        return [self.computable]

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
        self._child_sort()

    def _child_sort(self):
        pass

class AddNode(ArithmeticNode):
    symbol = '+'
    def _child_sort(self):
        child_keys = [child.get_key() for child in self.children]
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

    def as_argument_list(self, pattern):
        if not isinstance(pattern, kind.AddPatternNode):
            raise PatternMismatchError()
        if len(self.sorted_children) != len(pattern.sorted_children):
            raise PatternMismatchError()
        
        # Now, we need to apply the inverse permutation in `pattern`
        # to figure out how the function wants to be called and how our
        # own arguments match
        sorted_to_pattern = invert_permutation(pattern.child_permutation)
        arguments_in_sorted_order = [child.as_argument_list(pattern_child)
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

    def as_argument_list(self, pattern):
        if not isinstance(pattern, kind.MultiplyPatternNode):
            raise PatternMismatchError()
        if len(self.children) != len(pattern.children):
            raise PatternMismatchError()
        result = []
        for expr_child, pattern_child in zip(self.children, pattern.children):
            result.extend(expr_child.as_argument_list(pattern_child))
        return result

class SingleChildNode(ExpressionNode):
    pass

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
        self.dtype = child.dtype

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

    def as_argument_list(self, pattern):
        if not isinstance(pattern, kind.ConjugateTransposePatternNode):
            raise PatternMismatchError()
        return self.child.as_argument_list(pattern.child)

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
        self.dtype = child.dtype

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_inverse(*args, **kw)

    def as_argument_list(self, pattern):
        if not isinstance(pattern, kind.InversePatternNode):
            raise PatternMismatchError()
        return self.child.as_argument_list(pattern.child)



class BracketNode(ExpressionNode):
    symbol = 'b'
    
    def __init__(self, child, kinds=None):
        self.child = child
        self.children = [child]
        self.kinds = kinds
        self.nrows, self.ncols = child.nrows, child.ncols
        self.dtype = child.dtype

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_bracket(*args, **kw)

    def call_func(self, func, pattern):
        raise NotImplementedError()

for x, val in [
    (LeafNode, 1000),
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
