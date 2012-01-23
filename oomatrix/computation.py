import types

    

def register_computation(match, out_kind, obj):
    match.universe.add_computation(match, out_kind, obj)

def computation(match, out_kind):
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

        register_computation(match, out_kind, obj)
        return obj
            
    return dec
        
