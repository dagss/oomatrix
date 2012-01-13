class TODO:
    name = 'TODO'


class ExpressionNode(object):
    name = None

    def symbolic_add(self, other):
        return AddNode(self, other)

    def symbolic_mul(self, other):
        return MulNode(self, other)

    def get_type(self):
        return TODO
    
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

class DistributiveOperationNode(ExpressionNode):
    def __init__(self, a, b):
        self.children = children = []
        for x in (a, b):
            if type(x) is type(self):
                children.extend(x.children)
            else:
                children.append(x)
        # Following is correct both for multiplication and addition...
        self.nrows= children[0].nrows
        self.ncols = children[-1].ncols
        self.dtype = children[0].dtype # TODO combine better
    
class AddNode(DistributiveOperationNode):
    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_add(*args, **kw)

class MulNode(DistributiveOperationNode):
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

class InverseNode(ExpressionNode):
    def __init__(self, child):
        self.child = child
        self.children = [child]
        self.nrows, self.ncols = child.nrows, child.ncols
        self.dtype = child.dtype

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_inverse(*args, **kw)


for x, val in [
    (LeafNode, 1000),
    (InverseNode, 40),
    (ConjugateTransposeNode, 40),
    (MulNode, 30),
    (AddNode, 20)]:
    x.precedence = val

