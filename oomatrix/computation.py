import types
   

def register_computation(match, target_kind, obj):
    match.universe.join_with(target_kind.universe)
    match.universe.add_computation(match, target_kind, obj)

def register_conversion(from_kind, to_kind, obj):
    register_computation(from_kind, to_kind, obj)

def computation(match, target_kind):
    def dec(obj):
        # obj is expected to have a compute method; if not, assume it is
        # callable, and wrap it to provide it
        if not hasattr(obj, 'compute'):
            func = obj
            class Result(object):
                @staticmethod
                def compute(*args):
                    return func(*args)
                @staticmethod
                def cost(*args):
                    return 1
            Result.__name__ = obj.__name__
            Result.__module__ = obj.__module__
            obj = Result

        register_computation(match, target_kind, obj)
        return obj
    return dec


def conversion_method(target_kind):
    from .kind import add_post_class_definition_hook
    def dec(method):
        def doit(cls, method):
            return computation(cls, target_kind)(method)
        add_post_class_definition_hook(doit, method)
        return method
    return dec

def conversion(arg1, arg2=None):
    """Decorator for registering conversions
    
    A @conversion is really a computation from one kind to another::

        @computation(Diagonal, Dense)
        def diagonal_to_dense(x): ...

    However, we use a different decorator for readability purposes. Also,
    @conversion can be used on methods, and will in that case use the
    class as the first argument
    """
    if arg2 is None:
        return conversion_method(arg1)
    else:
        return computation(arg1, arg2)
        

