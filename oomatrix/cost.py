"""
Units for specifying cost of operations. For instance, a
dense matrix multiply may be

2 * m * n * FLOP + m * n * MEM

"""

class Cost(object):
    def __init__(self, **entries):
        self.entries = entries

    def __eq__(self, other):
        if not isinstance(other, Cost):
            return False
        return self.entries == other.entries

    def __ne__(self, other):
        return not self == other

    def __cmp__(self, other):
        raise Exception("Not supported, please use weigh method and compare resulting scalars")

    def __add__(self, other):
        if not isinstance(other, Cost):
            raise TypeError("Cannot add Cost with %s" % type(other))
        units = set(self.entries.keys() + other.entries.keys())
        result = {}
        for unit in units:
            result[unit] = self.entries.get(unit, 0) + other.entries.get(unit, 0)
        return Cost(**result)

    def __mul__(self, other):
        if isinstance(other, Cost):
            raise TypeError("Product of Cost with Cost not supported")
        result = dict(self.entries)
        for key in result:
            result[key] *= other
        return Cost(**result)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def weigh(self, **weights):
        """
        Yield a scalar value for the cost usable for comparison with other costs.

        E.g.::

            >>> Cost(FLOP=1, MEM=2).weigh(MEM=2, FLOP=1)
            5

        Parameters
        ----------
        weights : dict
           A dict with strings with unit names as keys, and unit weights
           as values.

        Returns
        -------
        weight : number
        """
        result = 0
        for unit, value in self.entries.iteritems():
            result += value * weights[unit]
        return result

    def __repr__(self):
        lst = list(self.entries.items())
        lst.sort()
        s = ' + '.join(['%s %s' % (value, unit) for unit, value in lst])
        return 'Cost(%s)' % s
        
FLOP = Cost(FLOP=1)
MEM = Cost(MEM=1)
MEMOP = Cost(MEMOP=1)
PRIORITY = Cost(PRIORITY=1)

default_cost_map = dict(
    FLOP=1,
    MEMOP=0.5,
    MEM=0,
    PRIORITY=1e-3)
