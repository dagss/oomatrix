



class ConversionCollection(object):
    pass

class BinaryOperationCollection(object):
    pass


class MatrixImpl(object):
    left_shape = None
    right_shape = None
    dtype = None

    def get_type(self):
        return type(self)

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

#class MatrixImplMetaclass(type):
#    def __init__(cls, name, bases, dct):
#        super(MatrixImplMetaclass, cls).__init__(name, bases, dct)
        
    
class TODO:
    name = 'todo'

