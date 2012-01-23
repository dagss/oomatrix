import types

def register_computation(match, out_kind, obj):
    match.universe.add_computation(match, out_kind, obj)

def computation(match, out_kind):
    def dec(obj):
        if isinstance(obj, (type, types.ClassType)):
            # new- or old-style class
            register_computation(match, out_kind, obj)
            return obj
    return dec
        
