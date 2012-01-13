
Storage format implementations
==============================


Operations
----------

Some operations are defined as "cheap".

E.g., `conjugate_transpose()` should not do any actual costly
transpose.  An actual in-memory transpose is done through conversion
operations, e.g.::

    @conversion(RowMajor.h, RowMajor)
    @conversion(RowMajor, RowMajor.h)
    def transpose(...)


a
