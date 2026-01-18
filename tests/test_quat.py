import functools
import operator

from math import isqrt, prod
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
    def assert_equal(res: Union[tuple, list, HurwitzQuaternion, hurwitzint], res_int: hurwitzint):
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
        r"""Test complexint \ complexint"""
        g = hurwitzint(1, 0, 0, 1)
        i = hurwitzint(0, 1, 0, 0)

        a = i * g

        q, r = a.rdivmod(g)
        assert not r
        assert g * q == a


class TestGcdLeft(HurwitzIntTests):
    """Tests for gcd_left"""

    @staticmethod
    def assert_left_divides(x: hurwitzint, g: hurwitzint):
        """Assert that g left-divides x (x = g*q, remainder 0 under right-division rdivmod)."""
        q, r = x.rdivmod(g)
        assert not r
        assert isinstance(q, hurwitzint)
        assert isinstance(r, hurwitzint)

    def test_zero(self):
        """gcd_left(a, 0) should return an associate of a (same norm) and left-divide a."""
        z = hurwitzint(0, 0, 0, 0)
        a = self.a_int

        d = a.gcd_left(z)

        self.assert_left_divides(a, d)
        assert abs(d) == abs(a)

    def test_recovers_constructed_common_factor(self):
        """gcd_left should recover a constructed common factor up to a unit (checked via norm)."""
        # Use units so we don't accidentally introduce extra common factors.
        i = hurwitzint(0, 1, 0, 0)
        j = hurwitzint(0, 0, 1, 0)

        # A small non-unit common left factor (norm 2 is the simplest).
        g = hurwitzint(1, 1, 0, 0)

        a = g * i
        b = g * j

        d = a.gcd_left(b)

        # d is a common left divisor
        self.assert_left_divides(a, d)
        self.assert_left_divides(b, d)

        # "Greatest": our known common divisor g must be a left multiple of d
        self.assert_left_divides(g, d)

        # If N(g) == N(d), then g = u*d for a unit u (so d matches g up to a unit).
        assert abs(d) == abs(g)


class TestGcdRight(HurwitzIntTests):
    """Tests for gcd_right"""

    @staticmethod
    def assert_right_divides(x: hurwitzint, g: hurwitzint):
        """Assert that g right-divides x (x = q*g, remainder 0 under left-division divmod)."""
        q, r = divmod(x, g)
        assert not r
        assert isinstance(q, hurwitzint)
        assert isinstance(r, hurwitzint)

    def test_zero(self):
        """gcd_right(a, 0) should return an associate of a (same norm) and right-divide a."""
        z = hurwitzint(0, 0, 0, 0)
        a = self.a_int

        d = hurwitzint.gcd_right(a, z)

        self.assert_right_divides(a, d)
        assert abs(d) == abs(a)

    def test_recovers_constructed_common_factor(self):
        """gcd_right should recover a constructed common factor up to a unit (checked via norm)."""
        # Use units so we don't accidentally introduce extra common factors.
        i = hurwitzint(0, 1, 0, 0)
        j = hurwitzint(0, 0, 1, 0)

        # A small non-unit common right factor (norm 2 is the simplest).
        g = hurwitzint(1, 1, 0, 0)

        a = i * g
        b = j * g

        d = a.gcd_right(b)

        # d is a common right divisor
        self.assert_right_divides(a, d)
        self.assert_right_divides(b, d)

        # "Greatest": our known common divisor g must be a right multiple of d
        self.assert_right_divides(g, d)

        # If N(g) == N(d), then g = u*d for a unit u (so d matches g up to a unit).
        assert abs(d) == abs(g)


class TestGcd(HurwitzIntTests):
    """Tests for gcd_left and gcd_right"""

    def test_gcd_agrees_with_integer_gcd_on_scalars(self):
        """For purely real scalars, gcd_left/gcd_right should match the integer gcd (up to sign/unit)."""
        a = hurwitzint(6, 0, 0, 0)
        b = hurwitzint(15, 0, 0, 0)

        dr = a.gcd_right(b)
        dl = a.gcd_left(b)

        # Scalar n has norm n^2, so sqrt(norm(gcd)) should recover gcd(|a|,|b|)=3
        assert isqrt(abs(dr)) == 3
        assert isqrt(abs(dl)) == 3

        # And the gcd should be purely real (imag parts 0)
        assert list(dr)[1:] == [0, 0, 0]
        assert list(dl)[1:] == [0, 0, 0]


class TestFactorRight(HurwitzIntTests):
    """Tests for factor_right"""

    def test_main(self):
        """Validate factor works as expected."""
        factors = self.b_int.factor_right()

        ans = prod(reversed(factors.primes), start=factors.unit)

        self.assert_equal(self.b_int, ans)


class TestFactorLeft(HurwitzIntTests):
    """Tests for factor_left"""

    def test_main(self):
        """Validate factor works as expected."""
        prod_left = lambda x, start=1: functools.reduce(operator.mul, x, 1) * start  # noqa: E731
        factors = self.b_int.factor_left()

        ans = prod_left(factors.primes, start=factors.unit)

        self.assert_equal(self.b_int, ans)
