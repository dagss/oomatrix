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

    def format_expression(self, name_to_matrix):
        raise NotImplementedError()
    
class LeafNode(ExpressionNode):
    def __init__(self, name, matrix_impl):
        self.name = name
        self.matrix_impl = matrix_impl
        self.nrows, self.ncols = matrix_impl.nrows, matrix_impl.ncols
        self.dtype = matrix_impl.dtype

    def get_type(self):
        return type(self.matrix_impl)

    def format_expression(self, name_to_matrix):
        try:
            x = name_to_matrix[self.name]
        except KeyError:
            name_to_matrix[self.name] = self.matrix_impl
        else:
            if self.matrix_impl is not x:
                raise NotImplementedError("Two matrices with same name")
        return self.name

class DistributiveOperationNode(ExpressionNode):
    def __init__(self, a, b):
        self.exprs = exprs = []
        for x in (a, b):
            if type(x) is type(self):
                exprs.extend(x.exprs)
            else:
                exprs.append(x)
        # Following is correct both for multiplication and addition...
        self.ncols = exprs[0].ncols
        self.nrows = exprs[-1].nrows
        self.dtype = exprs[0].dtype # TODO combine better
    
    def format_expression(self, name_to_matrix):
        exprs = [e.format_expression(name_to_matrix) for e in self.exprs]
        return self.infix_str.join(exprs)

class AddNode(DistributiveOperationNode):
    infix_str = ' + '

class MulNode(DistributiveOperationNode):
    infix_str = ' * '
