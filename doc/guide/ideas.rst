The idea of oomatrix
====================

``oomatrix`` is an idea for a Python library that should make it
possible to work in a high-level mode with numerical linear
algebra. It will achieve this by being an *object-oriented*,
*lazily-evaluated* library for numerical linear algebra. Let's see
what those mean in turn.

Object-oriented
---------------

The MATLAB and NumPy approach of thinking about matrices
as simply 2-dimensional arrays is fine for some things,
but not for others. When we say *matrix*, what we mean is
a linear operation, including:

 * Dense matrices (= 2D MATLAB and NumPy arrays)
 * Banded matrices, diagonal matrices, sparse matrices
 * Fourier transforms
 * Well, any linear function (perhaps implemented in Fortran etc.)

Now, it would obviously be nice to be able to express algorithms
in high-level form::

    V = U * (A + B + D).cholesky()

and not worry about the exact type/implementation of any of these.

Obviously, there's prior art for such a simple idea. So IMHO:

 * Sage (http://sagemath.org) implements this. My first idea was to
   improve this implementation. However it is not driven by numerical
   users, so there's not all that much to gain by starting here
   (beyond ideas and the API). For numerical uses it seems better
   to start on a piece that's isolated, rather than getting a huge
   dependency with no real gain.

 * PETSc (e.g., http://code.google.com/p/petsc4py/), Trilinos
   (http://trilinos.sandia.gov/). These are OO linear algebra for a
   cluster, with an emphasis on sparse matrices (came from PDE
   communities).  However the API is very little Pythonic and very
   "frameworky" -- a large part of the problem it solves simply
   doesn't exist in Python in the first place, and that part gets in
   your way and you have to write your program around it. Again useful
   to study, perhaps to use/wrap, but I don't feel they are starting
   points for use in non-PDE fields.

Neither of these are lazily evaluated.

Lazy evaluation
---------------

Or, if you like, "half-symbolic". To have proper object polymorphism
in this setting, I believe it is crucial to 


Details
-------

Some more points that can hopefully help get across the "feel":

 * There is no default vector type. Rather, one should hook
   the library up to other libraries that provide "vectors",
   such as NumPy. Multiplication of a matrix with a vector
   is *not* lazy but triggers evaluation::

       >>> a = np.linalg.normal(size=(16, 5, 3))
       >>> U = FourierTransformMatrix(16)
       >>> v = A * u # computes Fourier transform of 15 stacked vectors
       >>> type(v)
       <type 'numpy.ndarray'>
       >>> v.shape
       (16, 5, 3)

   The shape

 * The ``*`` operator is matrix multiplication

 * Matrices will be immutable until the right mutability mode can be
   sorted out. Mutating matrices is not very important.

