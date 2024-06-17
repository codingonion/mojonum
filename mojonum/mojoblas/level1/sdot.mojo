from tensor import Tensor

fn sdot(n: Int, sx: List[Float32], incx: Int, sy: List[Float32], incy: Int) -> Float32:
    """
    This function forms the dot product of two vectors,
    uses unrolled loops for increments equal to one.

    Arguments:
    n :
        Number of elements in the input vectors.
    sx:
        Real array, dimension (1 + (n - 1) * abs(incx))
    incx:
        Storage spacing between elements of sx.
    sy:
        Real array, dimension (1 + (n - 1) * abs(incy))
    incy:
        Storage spacing between elements of sy.

    Return:
        The dot product of two vectors.
    """
    var bs: Int = 1
    var stemp: Float32 = 0.0
    if n <= 0:
        return stemp
    if incx == 1 and incy == 1:
        # code for both increments equal to 1
        var m = n % 5
        if m != 0:
            for i in range(1, m + 1):
                stemp += sx[i - bs] * sy[i - bs]
            if n < 5:
                return stemp
        var mp1 = m + 1
        for i in range(mp1, n + 1, 5):
            stemp += (sx[i - bs]*sy[i - bs] +
                      sx[i - bs + 1]*sy[i - bs + 1] +
                      sx[i - bs + 2]*sy[i - bs + 2] +
                      sx[i - bs + 3]*sy[i - bs + 3] +
                      sx[i - bs + 4]*sy[i - bs + 4])
    else:
        # code for unequal increments or equal increments not equal to 1
        var ix: Int = 1
        var iy: Int = 1
        if incx < 0:
            ix = (-n + 1) * incx + 1
        if incy < 0:
            iy = (-n + 1) * incy + 1
        for _ in range(1, n + 1):
            stemp += sx[ix - bs] * sy[iy - bs]
            ix += incx
            iy += incy
    return stemp