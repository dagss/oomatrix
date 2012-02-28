"""
Units for specifying cost of operations. For instance, a
dense matrix multiply may be

2 * m * n * FLOP + m * n * MEM

"""

class CostValue(object):
    def __init__(self, **entries):
        self.entries = entries

    def __eq__(self, other):
        if not isinstance(other, CostValue):
            return False
        return self.entries == other.entries

    def __ne__(self, other):
        return not self == other

    def __cmp__(self, other):
        raise Exception("Not supported, please use weigh method and compare resulting scalars")

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, CostValue):
            raise TypeError("Cannot add CostValue with %s" % type(other))
        units = set(self.entries.keys() + other.entries.keys())
        result = {}
        for unit in units:
            result[unit] = self.entries.get(unit, 0) + other.entries.get(unit, 0)
        return CostValue(**result)

    def __mul__(self, other):
        if isinstance(other, CostValue):
            raise TypeError("Product of CostValue with CostValue not supported")
        result = dict(self.entries)
        for key in result:
            result[key] *= other
        return CostValue(**result)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def weigh(self, **weights):
        """
        Yield a scalar value for the cost usable for comparison with other costs.

        E.g.::

            >>> CostValue(FLOP=1, MEM=2).weigh(MEM=2, FLOP=1)
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
        return s
        
FLOP = CostValue(FLOP=1)
MEM = CostValue(MEM=1)
MEMOP = CostValue(MEMOP=1)
UGLY = CostValue(UGLY=1)

default_cost_map = dict(
    FLOP=1,
    MEMOP=0.5,
    MEM=0,
    UGLY=1e-3)
