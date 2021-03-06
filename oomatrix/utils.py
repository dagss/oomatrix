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

def cumsum(it):
    total = 0
    for x in it:
        total += x
        yield total

def complement_range(items, n):
    return [i for i in range(n) if i not in items]
            

