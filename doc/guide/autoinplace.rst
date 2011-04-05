Automatic in-place operations
=============================

There are often substantial benefits to doing an operation in-place
instead of making a copy. As an example, consider::

    (A + D).cholesky()

where A is a dense matrix and D is a diagonal matrix. If we are
allowed to overwrite the contents of A, adding the identity matrix is
a very quick operation; we simply increment the O(n) diagonal
elements, rather than copying all O(n**2) elements. Doing a Cholesky
decomposition in LAPACK in turn consumes no extra memory if we can
do the operation in-place and turn the array buffer of A into the
Cholesky factor.

The aim of oomatrix is to minimize the number of special cases in your
code. For maximum code reuse, you should not need to pass a flag
saying whether A can be overwritten or not, or even care if A and D
are dense or diagonal matrices. Thus we need to make the determination
of in-place operations automatic.

Deleting unused objects
-----------------------

The most important thing to remember is to delete unused objects
before computing. Consider this code::

    v = (A + D).cholesky() * u

where u is a NumPy array, forcing computation.  Because one still
hold references to A and D and they can be used for other purposes
later, in-place operations are not possible. Instead, one should do::

    L = (A + D).cholesky() # lazy, not computed
    del A; del D; # or A = D = None
    v = L * u # expression evaluated here

Since it is no longer possible to use A at the point the computation
actually triggers, this leads to in-place operations being done --
except if other parts of the program also holds references to A.

*Implementation note:* This can be implemented simply by counting
references from Matrix to DenseMatrixStore, and does not depend on
CPython reference counting, so this should be simple to work out.

Giving up objects
-----------------

A trickier case occurs when one wants to do in-place operations on
matrices passed between functions. Consider::

    def f(A, D, u):
        L = (A + D).cholesky()
        del A; del D;
        return L * u

Even with ``del`` statements, it may be the case that the calling code
holds on to the matrices -- or not. This is where the ``give``
function comes into play. By calling ``give``, you promise to never
use the same variable again within the same function. If this seems
complicated, remember that it is always safe to leave ``give`` out --
it is only a measure to reduce CPU-time and memory use.

Functions that are to recieve given objects need to be decorated with
``@takes()``.  Example::

    @takes()
    def compute_v(M1, M2, u):
        L = (M1 + M2).cholesky()
        del M1; del M2;
        return L * u

    @takes()
    def compute_both_v(X, Y1, Y2, u1, u2):
        v1 = compute_v(X, give(Y1), u1)
        v2 = compute_v(give(X), give(Y2), u2)
        return v1, v2

    # Assume A is dense, D1 and D2 diagonal, u1 and u2 arrays
    v1, v2 = compute_both_v(give(A), give(D1), give(D2), u1, u2)
    

In this case, the first time ``compute_v`` is called, a copy is made
of the dense matrix A, while the second time, A can safely be
overwritten. On the other hand, consider changing the top-level
call to::

    v1, v2 = compute_both_v(D1, give(A), give(D2), u1, u2)

Now, A will be overwritten the first time ``compute_v`` is called,
while the second time it is determined that D1 can be added in-place
to D2.

To trigger an in-place operation, there needs to be a full chain of
``give`` calls from the top-level call to the bottom.  Note that
``give`` relies on the number of references to the same matrix in the
calling scope::

    @takes()
    def f(A):
        B = A
        func(give(A)) # no chance of in-place operation
        ...


Here, the existence of the B reference at the time of calling ``give``
prevents in-place operations -- you are allowed to continue using
the B reference, and so the matrix contents is not overwritten.

Note that if you pass the same argument twice you should only give it
once. Thus it is::

    func(A, give(A))
    # or: func(give(A), A)

But (!)::

    B = A
    func(give(A), give(B))

The need for ``give`` can often be avoided by returning matrix
expressions rather than computed values. Some of the above code can
be refactored like this::

    @takes() # as a courtesy, in case caller unecesarrily calls give
    def compute_L(M1, M2):
        return (M1 + M2).cholesky()

    @takes()
    def compute_both_v(X, Y1, Y2, u1, u2):
        L = compute_L(X, Y1)
        del Y1
        v1 = L * u1
        del L
        L = compute_L(X, Y2)
        del X; del Y2;
        v2 = L * u2
        return v1, v2


Safety measures
---------------

What happens if you do not honor your part of the contract? I.e.::

    func(give(X))
    print X # you promised not to do this...

Essentially, if the underlying buffer of X was used for in-place
operations in the end, the X object will have been destructed,
and you get::

    <destructed matrix (invalid use of give)>

Otherwise, unfortunately, nothing has happened (since the object X
could be available in other parts of the code as well, we can't just
change it).

Another scenario is if you decide to use ``give`` itself in strange
ways, such as::

    def f(x):
        print x
    f(give(A))

Here the problem is that ``f`` lacks the ``@takes()`` decorator.  It
will then see ``x`` as a ``DeletionPromise`` object, and nothing much
else happens. The only way ``A`` will actually get destructed is by
having a sufficient number of ``DeletionPromise`` objects, and passing
it through the ``@takes()`` decorator. The task of the ``@takes()``
decorator is to unwrap matrices from the ``DeletionPromise`` objects,
so normally you do not see them.



How does it work?
-----------------

I only scetch the "inner" matter of actually figuring out the in-place
operations. This will surely work by using some weak-references: If a
``Matrix`` has a strong reference to a ``DenseMatrixStore``, then the
latter can put the former in a list of weak references to track the
number of references to it. Or, override ``__del__`` and use manual
reference counting.

The tricky part is the giving away bit, where we are asked to disregard
a given ``Matrix`` even if it is held on to:

 1. When ``give(A)`` is called, the ``give`` function decides how many
    references A has got in the calling scope. The reference count at the
    point of the call is used raw, except in the one case where A was
    constructed by the wrapping ``@takes`` (which sets a flag in the 
    matrix), in which case one is subtracted.

    Note that since we "capture" the refcount at the point of calling
    ``give``, it is much easier to control the context. We don't have
    to worry about the call stack passing through some C code with
    strange refcount semantics, as long as ``give`` itself is called
    from normal Python or Cython code.

 2. ``give(A)`` always returns a ``DeletionPromise``, which contain a
    reference to A plus a ``required_count`` which specifices how many
    variables referring to A would need to be deleted to deallocate A.
    Examples::

        A = B = identity_matrix(10)
        f(A, give(B)) # One DeletionPromise passed with count == 2.

        A = B = identity_matrix(10)
        f(give(A), give(B)) # Two DeletionPromise passed, each with count == 2

        A = B = othermodule.matrix
        f(give(A), give(B)) # Two DeletionPromise passed, each with count > 2

 3. The ``DeletionPromise`` can now travel safely as the user is not
    allowed to fetch the wrapped matrix reference. Either right away
    or after passing through some other function (which only forwards
    arguments), it hits a ``takes`` decorator (otherwise, the user
    will just get unexpected ``DeletionPromise`` objects, which will
    have a nice help text in their ``repr``). The ``takes`` looks at
    all arguments passed in, raises exceptions if it sees anything
    strange, and for good measure destructs the ``DeletionPromise``
    before returning. But the important part is:

    Case I: For a given wrapped matrix, the ``required_count`` of each
    corresponding ``DeletionPromise`` matches the number of
    ``DeletionPromise`` given for the matrix. In that case, a) make a
    destructive copy, so that we have a new object for the matrix and
    destruct the object in the caller scope, b) mark our new matrix
    so that our own reference to it is not taken into account in
    ``give``, c) call the decorated function, d) remove the mark. The
    callee now has a new object, the caller a destructed one.

    Case II: Otherwise, too many references are held somewhere in the
    caller stack. Simply pass on the inner matrix and free the
    ``DeletionPromise``. The caller and callee now share a common
    object with refcount so high it is never considered for inplace
    until we return.

 4. Finally, our matrix enters the symbolic tree which is asked to
    compute itself. If the only reference to the inner, say,
    ``DenseMatrixStore`` is through a matrix that has reference count
    1 and is marked as being constructed by ``takes``, then go for
    inplace -- the only reference being held is within the ``@takes``.

To argue that this is correct, consider a function decorated by ``@takes``
and calling another function by using ``give``:

 * Consider each level of the parent call stack which is *not* a call
   from ``@takes`` to its decorated function. Either a ``DeletionPromise``
   is passed, or a matrix that is not given away is passed. Either way,
   we do not care about the reference count of the arguments.

 * In ``give``, the matrix considered will either a) be passed in to
   the function without being given away (this includes 3-II above),
   b) be fetched by the function, or c) be constructed by the wrapping
   ``@takes``. In case a) the reference count is so high that we don't
   care (assuming the calling code is Cython or Python code), the
   ``DeletionPromise`` taken will be discarded. In case b) the
   refcount can be used directly, while in case c) there's a flag and
   we can adjust the refcount directly.



