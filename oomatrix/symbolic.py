class TODO:
    name = 'TODO'


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
    

class LeafNode(ExpressionNode):
    children = ()
    
    def __init__(self, name, matrix_impl):
        self.name = name
        self.matrix_impl = matrix_impl
        self.nrows, self.ncols = matrix_impl.nrows, matrix_impl.ncols
        self.dtype = matrix_impl.dtype

    def get_type(self):
        return type(self.matrix_impl)

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_leaf(*args, **kw)

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
    
class AddNode(ArithmeticNode):
    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_add(*args, **kw)

    def can_distribute(self):
        return True
    
    def distribute_right(self, other):
        return AddNode([MulNode([term, other])
                        for term in self.children])

    def distribute_left(self, other):
        return AddNode([MulNode([other, term])
                        for term in self.children])

class MulNode(ArithmeticNode):
    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_multiply(*args, **kw)

class ConjugateTransposeNode(ExpressionNode):
    def __init__(self, child):
        assert not isinstance(child, ConjugateTransposeNode), 'Double conjugation not allowed'
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
        return AddNode([MulNode([term.conjugate_transpose(), other])
                        for term in terms])

    def distribute_left(self, other):
        if not isinstance(self.child, AddNode):
            raise AssertionError()
        terms = self.child.children
        return AddNode([MulNode([other, term.conjugate_transpose()])
                        for term in terms])

    def conjugate_transpose(self):
        return self.child

class InverseNode(ExpressionNode):
    def __init__(self, child):
        self.child = child
        self.children = [child]
        self.nrows, self.ncols = child.nrows, child.ncols
        self.dtype = child.dtype

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_inverse(*args, **kw)

class BracketNode(ExpressionNode):
    def __init__(self, child, kinds=None):
        self.child = child
        self.children = [child]
        self.kinds = kinds
        self.nrows, self.ncols = child.nrows, child.ncols
        self.dtype = child.dtype

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_bracket(*args, **kw)


for x, val in [
    (LeafNode, 1000),
    (BracketNode, 1000),
    (InverseNode, 40),
    (ConjugateTransposeNode, 40),
    (MulNode, 30),
    (AddNode, 20)]:
    x.precedence = val



#
# Symbolic tree manipulation tools
#

def apply_right_distributive_rule(expr):
    """
    Applies the right-distributive rule on a MulNode.

    E.g., the right distributive rule yields::

      (A + B) * x => A * x + B * x
      (A + B).h * x => A.h * x + B.h * x
      A * (C + D) => A * (C + D) # not left-distributive

      A * (B + C) * (D + E) * x =>
        A * (B * (D * x + E * x) + C * (D * x + E * x)) 
    
    
    """
    if not isinstance(expr, MulNode):
        raise TypeError('expr must be MulNode')
    head, tail = expr.children[0], expr.children[1:]
    # Recurse to distribute on tail
    if len(tail) == 1:
        processed_tail = tail[0]
    else:
        processed_tail = apply_right_distributive_rule(MulNode(tail))
    if head.can_distribute():
        return head.distribute_right(processed_tail)
    else:
        return MulNode([head, processed_tail])

def apply_left_distributive_rule(expr):
    """
    Applies the right-distributive rule on a MulNode.
    """
    if not isinstance(expr, MulNode):
        raise TypeError('expr must be MulNode')
    tail, head = expr.children[:-1], expr.children[-1]
    # Recurse to distribute on tail
    if len(tail) == 1:
        processed_tail = tail[0]
    else:
        processed_tail = apply_left_distributive_rule(MulNode(tail))
    if head.can_distribute():
        return head.distribute_left(processed_tail)
    else:
        return MulNode([processed_tail, head])
