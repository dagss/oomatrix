import heapq

class Heap(object):
    """
    Wraps heapq to provide a heap. The sorting is done by provided cost
    and then insertion order; the values are never compared.
    """

    def __init__(self):
        self.heap = []
        self.order = 0

    def push(self, cost, value):
        x = (cost, self.order, value)
        self.order += 1
        heapq.heappush(self.heap, x)

    def pop(self):
        cost, order, value = heapq.heappop(self.heap)
        return cost, value

    
