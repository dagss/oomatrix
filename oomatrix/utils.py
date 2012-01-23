def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def invert_permutation(permutation):
    result = [None] * len(permutation)
    for ri, p in enumerate(permutation):
        result[p] = ri
    return result

