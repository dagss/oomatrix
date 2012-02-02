

In general, operations that have only a very small overhead that is
independent of the matrix size happens at once, while other operations
are left until `compute()` is called. So, e.g., ``3 * A * 3`` will
immediately turn into ``9 * A``, but one has to call `compute()` in
order to actually multiply ``9`` into each element of ``A``.

Conjugate transpose, ``A.h``, happens immediately in the sense that if
``A`` is computed, then ``A.h`` is also able to support element
lookups, `diagonal()`, `as_array()` etc. without further computation.
However the same underlying representation is used (unless a zero-cost
conversion is registered) [not implemented]
