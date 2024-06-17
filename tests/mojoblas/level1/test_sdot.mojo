from testing import assert_almost_equal

import mojonum as mn

fn test_sdot_with_basic() raises:
    """Test sdot with basic case where both vectors have unit increment."""
    var n = 5
    var sx = List[Float32](1, 2, 3, 4, 5)
    var sy = List[Float32](5, 4, 3, 2, 1)
    var result = mn.sdot(n, sx, 1, sy, 1)
    assert_almost_equal(result, 35.0, msg = "Expected 35.0 but got " + str(result))

fn test_sdot_with_zero_length() raises:
    """Test sdot with zero length vectors."""
    var n = 0
    var sx = List[Float32]()
    var sy = List[Float32]()
    var result = mn.sdot(n, sx, 1, sy, 1)
    assert_almost_equal(result, 0.0, msg = "Dot product should be 0.0 for empty vectors")

fn test_sdot_non_unit_increments() raises:
    """Test sdot with non-unit increments."""
    var n = 4
    var sx = List[Float32](1, 3, 5, 7, 9, 11, 13, 15)
    var sy = List[Float32](2, 4, 6, 8, 10, 12, 14, 16)
    var result = mn.sdot(n, sx, 2, sy, 2)
    assert_almost_equal(result, 304.0, msg = "Expected 304.0 but got " + str(result))

fn test_sdot_with_negative_increments() raises:
    """Test sdot with negative increments."""
    var n = 3
    var sx = List[Float32](1, 2, 3)
    var sy = List[Float32](3, 2, 1)
    var result = mn.sdot(n, sx, -1, sy, -1)
    assert_almost_equal(result, 10.0, msg = "Expected 10.0 but got " + str(result))