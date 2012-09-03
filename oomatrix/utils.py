def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def invert_permutation(permutation):
    result = [None] * len(permutation)
    for ri, p in enumerate(permutation):
        result[p] = ri
    return result

def sort_by(items, keys, reverse=False):
    zipped = zip(keys, items)
    zipped.sort(reverse=reverse)
    return [second for first, second in zipped]

