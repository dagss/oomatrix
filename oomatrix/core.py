



class ConversionCollection(object):
    pass

class BinaryOperationCollection(object):
    pass


class MatrixRepresentation(object):
    left_shape = None
    right_shape = None


    def symbolic_add(self, other):
        return AddedMatrices([self, other])

    def symbolic_mul(self, other):
        return MultipliedMatrices([self, other])

    #TODO: Nice idea, not needed now
    ## def memory_use(self):
    ##     """
    ##     Probe the memory use of the stored matrix

    ##     Returns
    ##     -------
    ##     size_and_objs : list of (int, object)
    ##         List of objects that holds on to the memory, and
    ##         The idea is that objects passed to the constructor can
    ##         be shared, and the object reference can be used to
    ##         identify data sharing.
    ##     """
    ##     raise NotImplementedError()
    ##     return (self.array, self.array.size * self.array.itemsize)
        

class AddAction(object):
    pass


def conversion(target_cls):
    def dec(func):
        return func
    return dec

def conversion_cost(target_cls):
    def dec(func):
        return func
    return dec

#class MatrixRepresentationMetaclass(type):
#    def __init__(cls, name, bases, dct):
#        super(MatrixRepresentationMetaclass, cls).__init__(name, bases, dct)
        
    


class ExpressionNode(MatrixRepresentation):
    pass

class AddedMatrices(ExpressionNode):
    def __init__(self, matrices):
        self.matrices = matrices
        if len(matrices) == 0:
            raise ValueError()
        self.left_shape = matrices[0].left_shape
        self.right_shape = matrices[0].right_shape

    def symbolic_add(self, other):
        if isinstance(other, AddedMatrices):
            return AddedMatrices(self.matrices + other.matrices)
        else:
            return AddedMatrices(self.matrices + [other])


class MultipliedMatrices(ExpressionNode):
    def __init__(self, matrices):
        self.matrices = matrices
        if len(matrices) == 0:
            raise ValueError()
        self.left_shape = matrices[0].left_shape
        self.right_shape = matrices[-1].right_shape

    def symbolic_mul(self, other):
        if isinstance(other, MultipliedMatrices):
            return MultipliedMatrices(self.matrices + other.matrices)
        else:
            return MultipliedMatrices(self.matrices + [other])

