from pathlib import Path
from typing import Union

from hurwitz import HurwitzQuaternion

import quatint.quat

from quatint.quat import hurwitzint

def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of hurwitzint"""
    path = Path(quatint.quat.__file__)
    assert path.suffix.lower() != '.py'


def test_is_instance():
    """Verify that basic isinstance checks work"""
    assert isinstance(hurwitzint(1, 2, 3, 4), hurwitzint)
    assert not isinstance(complex(1, 2), hurwitzint)


class HurwitzIntTests:
    """Support methods for testing hurwitzint"""
    a, b, a_int, b_int = None, None, None, None

    def setup_method(self, _):
        """Setup some test data"""
        self.a = HurwitzQuaternion(1, 2, 3, 4)
        self.b = HurwitzQuaternion(2, 3, 4, 5)

        self.a_int = hurwitzint(1, 2, 3, 4)
        self.b_int = hurwitzint(2, 3, 4, 5)

    @staticmethod
    def assert_equal(res: Union[tuple, list, HurwitzQuaternion], res_int: hurwitzint):
        """Validate the hurwitzint is equal to the validation object, and that it is still backed by integers"""
        if isinstance(res, HurwitzQuaternion):
            res = [x * 2 for x in res]

        assert list(res) == list(res_int)

        assert isinstance(res_int.a, int)
        assert isinstance(res_int.b, int)
        assert isinstance(res_int.c, int)
        assert isinstance(res_int.d, int)

        assert isinstance(res_int, hurwitzint)


class TestEq(HurwitzIntTests):
    """Tests for __eq__"""

    def test_main(self):
        """Basic equals tests"""
        c = hurwitzint(1, 2, 3, 4)
        assert self.a_int == c
        assert self.b_int != c


class TestAdd(HurwitzIntTests):
    """Tests for __add__"""

    def test_add(self):
        """Test hurwitzint + hurwitzint"""
        res = self.a + self.b
        res_int = self.a_int + self.b_int

        self.assert_equal(res, res_int)

    def test_add_int(self):
        """Test hurwitzint + int"""
        for i in range(100):
            res_int = self.a_int + i

            self.assert_equal((2 + i * 2, 4, 6, 8), res_int)

    def test_add_int_reversed(self):
        """Test int + hurwitzint"""
        for i in range(100):
            res_int = i + self.a_int

            self.assert_equal((2 + i * 2, 4, 6, 8), res_int)

    def test_add_float(self):
        """Test hurwitzint + float"""
        for i in range(100):
            res_int = self.a_int + float(i)

            self.assert_equal((2 + i * 2, 4, 6, 8), res_int)

    def test_add_float_reversed(self):
        """Test float + hurwitzint"""
        for i in range(100):
            res_int = float(i) + self.a_int

            self.assert_equal((2 + i * 2, 4, 6, 8), res_int)


class TestSub(HurwitzIntTests):
    """Tests for __sub__"""

    def test_sub(self):
        """Test hurwitzint - hurwitzint"""
        res = self.a - self.b
        res_int = self.a_int - self.b_int

        self.assert_equal(res, res_int)

    def test_sub_int(self):
        """Test hurwitzint - int"""
        for i in range(100):
            res_int = self.a_int - i

            self.assert_equal((2 - i * 2, 4, 6, 8), res_int)

    def test_sub_int_reversed(self):
        """Test int - hurwitzint"""
        for i in range(100):
            res_int = i - self.a_int

            self.assert_equal((i * 2 - 2, -4, -6, -8), res_int)

    def test_sub_float(self):
        """Test hurwitzint - float"""
        for i in range(100):
            res_int = self.a_int - float(i)

            self.assert_equal((2 - i * 2, 4, 6, 8), res_int)

    def test_sub_float_reversed(self):
        """Test float - hurwitzint"""
        for i in range(100):
            res_int = float(i) - self.a_int

            self.assert_equal((i * 2 - 2, -4, -6, -8), res_int)


class TestNegPos(HurwitzIntTests):
    """Tests for __neg__ and __pos__"""

    def test_neg(self):
        """Test -hurwitzint"""
        res = -self.a
        res_int = -self.a_int

        self.assert_equal(res, res_int)

    def test_pos(self):
        """Test +hurwitzint"""
        res_int = +self.a_int

        self.assert_equal((2, 4, 6, 8), res_int)


class TestMul(HurwitzIntTests):
    """Tests for __mul__"""

    def test_mul(self):
        """Test hurwitzint * hurwitzint"""
        # Also test that this operation is non-commutative
        res = self.a * self.b
        res_int1 = self.a_int * self.b_int
        self.assert_equal(res, res_int1)

        res = self.b * self.a
        res_int2 = self.b_int * self.a_int
        self.assert_equal(res, res_int2)

        assert res_int1 != res_int2

    def test_mul_int(self):
        """Test hurwitzint * int"""
        for i in range(100):
            res_int = self.a_int * i

            self.assert_equal((2 * i, 4 * i, 6 * i, 8 * i), res_int)

    def test_mul_int_reversed(self):
        """Test int * hurwitzint"""
        for i in range(100):
            res_int = i * self.a_int

            self.assert_equal((2 * i, 4 * i, 6 * i, 8 * i), res_int)

    def test_mul_float(self):
        """Test hurwitzint * float"""
        for i in range(100):
            res_int = self.a_int * float(i)

            self.assert_equal((2 * i, 4 * i, 6 * i, 8 * i), res_int)

    def test_mul_float_reversed(self):
        """Test float * complexint"""
        for i in range(100):
            res_int = float(i) * self.a_int

            self.assert_equal((2 * i, 4 * i, 6 * i, 8 * i), res_int)


class TestDiv(HurwitzIntTests):
    """Tests for __truediv__ and __floordiv__"""

    def test_div(self):
        """Test complexint / complexint"""
        res_q, res_r = self.a.euclidean_division(self.b)
        res_int_q, res_int_r = divmod(self.a_int, self.b_int)

        self.assert_equal(res_q, res_int_q)
        self.assert_equal(res_r, res_int_r)


class TestRDiv(HurwitzIntTests):
    """Tests for rtruediv and rfloordiv"""

    def test_rdiv(self):
        """Test complexint \ complexint"""
        g = hurwitzint(1, 0, 0, 1)
        i = hurwitzint(0, 1, 0, 0)

        a = i * g

        q, r = a.rdivmod(g)
        assert not r
        assert g * q == a
