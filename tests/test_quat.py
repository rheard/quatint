import os

from math import isqrt
from pathlib import Path
from typing import Union

import pytest

from hurwitz import HurwitzQuaternion

import quatint.quat

from quatint.quat import NonCommutativeFactorization, hurwitzint, rdivmod, prod_left, prod_right

@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"},
                    reason="Compiled-only test")
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

    def test_wide_search(self):
        """Test many different hurwitzint division operations"""
        a_max = 7
        b_max = 3

        for a1 in range(1, a_max):
            for b1 in range(1, a_max):
                for c1 in range(1, a_max):
                    for d1 in range(1, a_max):
                        for a2 in range(1, b_max):
                            for b2 in range(1, b_max):
                                for c2 in range(1, b_max):
                                    for d2 in range(1, b_max):
                                        a = hurwitzint(a1, b1, c1, d1)
                                        b = hurwitzint(a2, b2, c2, d2)

                                        res_q, res_r = divmod(a, b)

                                        assert res_q * b + res_r == a


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

    def test_wide_search(self):
        """Test many different hurwitzint right-division operations"""
        a_max = 7
        b_max = 3

        for a1 in range(1, a_max):
            for b1 in range(1, a_max):
                for c1 in range(1, a_max):
                    for d1 in range(1, a_max):
                        for a2 in range(1, b_max):
                            for b2 in range(1, b_max):
                                for c2 in range(1, b_max):
                                    for d2 in range(1, b_max):
                                        a = hurwitzint(a1, b1, c1, d1)
                                        b = hurwitzint(a2, b2, c2, d2)

                                        res_q, res_r = rdivmod(a, b)

                                        assert b * res_q + res_r == a


class TestIsUnit(HurwitzIntTests):
    """Tests for is_unit"""

    def test_units(self):
        """Validate all known Hurwitz units are detected as units."""
        assert len(hurwitzint.UNITS) == 24

        for unit in hurwitzint.UNITS:
            assert unit.is_unit
            assert abs(unit) == 1

    def test_non_units(self):
        """Validate non-units are not detected as units."""
        for n in (
            hurwitzint(0, 0, 0, 0),
            hurwitzint(2, 0, 0, 0),
            hurwitzint(1, 1, 0, 0),
            hurwitzint(1, 2, 3, 4),
            hurwitzint(3, 1, 1, 1, half=True),
        ):
            assert not n.is_unit

    def test_half_units(self):
        """Validate true half-integer Hurwitz units are detected as units."""
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    for d in (-1, 1):
                        unit = hurwitzint(a, b, c, d, half=True)

                        assert unit.is_unit
                        assert abs(unit) == 1


class TestInverse(HurwitzIntTests):
    """Tests for inverse"""

    def test_units_inverse_by_multiplication(self):
        """Validate every unit inverse multiplies back to one on both sides."""
        one = hurwitzint(1, 0, 0, 0)

        for unit in hurwitzint.UNITS:
            inv = unit.inverse()

            assert isinstance(inv, hurwitzint)
            assert inv.is_unit
            assert unit * inv == one
            assert inv * unit == one

    def test_units_inverse_is_conjugate(self):
        """Validate the inverse of a Hurwitz unit is its conjugate."""
        for unit in hurwitzint.UNITS:
            assert unit.inverse() == unit.conjugate()

    def test_inverse_of_inverse(self):
        """Validate taking the inverse twice recovers the original unit."""
        for unit in hurwitzint.UNITS:
            assert unit.inverse().inverse() == unit

    def test_non_unit_inverse_raises(self):
        """Validate non-units do not have inverses in the Hurwitz integers."""
        for n in (
            hurwitzint(0, 0, 0, 0),
            hurwitzint(2, 0, 0, 0),
            hurwitzint(1, 1, 0, 0),
            hurwitzint(1, 2, 3, 4),
        ):
            with pytest.raises(ValueError):
                n.inverse()

    def test_negative_power_for_units_if_supported(self):
        """Validate negative powers of units agree with inverse powers."""
        i = hurwitzint(0, 1, 0, 0)

        try:
            res = i ** -1
        except ValueError:
            pytest.skip("Negative powers are not supported")
        else:
            assert res == i.inverse()
            assert i ** -2 == i.inverse() * i.inverse()


class TestSplitLipschitz(HurwitzIntTests):
    """Tests for split_lipschitz"""

    def test_lipschitz_integer_returns_self_and_none(self):
        """Validate Lipschitz/integer quaternions do not require a half-unit part."""
        for n in (
            hurwitzint(0, 0, 0, 0),
            hurwitzint(1, 2, 3, 4),
            hurwitzint(-1, -2, -3, -4),
            hurwitzint(5, 0, -2, 7),
        ):
            whole, half = n.split_lipschitz()

            assert whole == n
            assert half is None

    def test_half_integer_splits_into_whole_plus_half_unit(self):
        """Validate true Hurwitz half-integers split into a Lipschitz part plus one half-unit."""
        examples = (
            hurwitzint(3, 5, 7, 9, half=True),
            hurwitzint(-3, 5, -7, 9, half=True),
            hurwitzint(1, 1, 1, 1, half=True),
            hurwitzint(-1, -1, -1, -1, half=True),
        )

        for n in examples:
            whole, half = n.split_lipschitz()

            assert isinstance(whole, hurwitzint)
            assert isinstance(half, hurwitzint)

            assert whole.is_lipschitz
            assert half.is_unit
            assert not half.is_lipschitz

            assert whole + half == n

    def test_split_examples(self):
        """Validate split_lipschitz returns the expected whole and half-unit parts."""
        n = hurwitzint(3, 5, 7, 9, half=True)

        whole, half = n.split_lipschitz()

        self.assert_equal((2, 4, 6, 8), whole)
        self.assert_equal((1, 1, 1, 1), half)
        assert whole + half == n

        n = hurwitzint(-3, 5, -7, 9, half=True)

        whole, half = n.split_lipschitz()

        self.assert_equal((-2, 4, -6, 8), whole)
        self.assert_equal((-1, 1, -1, 1), half)
        assert whole + half == n

    def test_split_half_unit(self):
        """Validate a half-unit splits into zero plus itself."""
        n = hurwitzint(1, -1, 1, -1, half=True)

        whole, half = n.split_lipschitz()

        assert whole == hurwitzint(0, 0, 0, 0)
        assert half == n
        assert whole + half == n

    def test_split_reconstructs_many_half_integers(self):
        """Validate split_lipschitz reconstructs many true Hurwitz half-integers."""
        for a in range(-9, 10, 2):
            for b in range(-9, 10, 2):
                for c in range(-9, 10, 2):
                    for d in range(-9, 10, 2):
                        n = hurwitzint(a, b, c, d, half=True)

                        whole, half = n.split_lipschitz()

                        assert whole.is_lipschitz
                        assert half is not None
                        assert half.is_unit
                        assert whole + half == n


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

        assert dr.a == 6
        assert dl.a == 6


class TestFactorRightDetail(HurwitzIntTests):
    """Tests for factor_right_detail"""

    def test_main(self):
        """Validate factor works as expected."""
        self.assert_factoring(self.b_int, self.b_int.factor_right_detail())

    def test_examples(self):
        """Validate factor works as expected for some given examples."""
        n = hurwitzint(2, 3, 4, 53)
        self.assert_factoring(n, n.factor_right_detail())

        # This fails to have a norm-sorted prime factorization if metacommutation has not been implimented
        n = hurwitzint(1, 1, 1, 6)
        self.assert_factoring(n, n.factor_right_detail())

        # This fails to factor after metacommutation was implemented due to a failed metacommutation swap. Fix it!
        n = hurwitzint(1, 1, 2, 15)
        self.assert_factoring(n, n.factor_right_detail())

        n = hurwitzint(17 * 31, 0, 0, 0)
        self.assert_factoring(n, n.factor_right_detail())

    def assert_factoring(self, n: hurwitzint, factors: NonCommutativeFactorization):
        """Validate everything about the factoring is correct"""
        ans = factors.prod_right()

        self.assert_equal(n, ans)

        # Validate metacommutation by verifying the norms are sorted
        norms = [abs(p) for p in factors.primes]
        assert norms == sorted(norms)

        for p in factors.primes:
            # These _should_ all be primes and should be impossible to factor...
            prime_factors = p.factor_right_detail()

            assert prime_factors.content == 1
            assert abs(prime_factors.unit) == 1
            assert len(prime_factors.primes) == 1
            assert abs(prime_factors.primes[0]) == abs(p)

            q, r = divmod(p, prime_factors.primes[0])
            assert not r
            assert abs(q) == 1


class TestFactorRight(HurwitzIntTests):
    """Tests for factor_right"""

    def test_main(self):
        """Validate factor_right returns factors whose product is the original number."""
        factors = self.b_int.factor_right()

        ans = prod_right(factors)

        self.assert_equal(self.b_int, ans)

    def test_examples(self):
        """Validate factor_right works as expected for some given examples."""
        for n in (
            hurwitzint(2, 3, 4, 53),
            hurwitzint(1, 1, 1, 6),
            hurwitzint(1, 1, 2, 15),
            hurwitzint(17 * 31, 0, 0, 0),
        ):
            factors = n.factor_right()

            ans = prod_right(factors)

            self.assert_equal(n, ans)

    def test_with_content_and_multiple_factors(self):
        """Validate factor_right does not apply scalar content more than once."""
        n = hurwitzint(6, 2, 4, 0)

        factors = n.factor_right(canonical=False)

        assert n.factor_right_detail(canonical=False).content > 1
        assert len(n.factor_right_detail(canonical=False).primes) > 1

        ans = prod_right(factors)

        self.assert_equal(n, ans)

    def test_canonical_with_content_and_multiple_factors(self):
        """Validate canonical factor_right works when scalar content is present."""
        n = hurwitzint(6, 2, 4, 0)

        factors = n.factor_right(canonical=True)

        assert n.factor_right_detail(canonical=True).content > 1
        assert len(n.factor_right_detail(canonical=True).primes) > 1

        ans = prod_right(factors)

        self.assert_equal(n, ans)


class TestFactorLeftDetail(HurwitzIntTests):
    """Tests for factor_left_detail"""

    def test_main(self):
        """Validate factor works as expected."""
        self.assert_factoring(self.b_int, self.b_int.factor_left_detail())

    def test_examples(self):
        """Validate factor works as expected for some given examples."""
        n = hurwitzint(2, 3, 4, 53)
        self.assert_factoring(n, n.factor_left_detail())

        # This fails to have a norm-sorted prime factorization if metacommutation has not been implimented
        n = hurwitzint(1, 1, 1, 6)
        self.assert_factoring(n, n.factor_left_detail())

        # This fails to factor after metacommutation was implemented due to a failed metacommutation swap. Fix it!
        n = hurwitzint(1, 1, 2, 15)
        self.assert_factoring(n, n.factor_left_detail())

        n = hurwitzint(17 * 31, 0, 0, 0)
        self.assert_factoring(n, n.factor_left_detail())

    def assert_factoring(self, n: hurwitzint, factors: NonCommutativeFactorization):
        """Validate everything about the factoring is correct"""
        ans = factors.prod_left()

        self.assert_equal(n, ans)

        # Validate metacommutation by verifying the norms are sorted
        norms = [abs(p) for p in factors.primes]
        assert norms == sorted(norms)

        for p in factors.primes:
            # These _should_ all be primes and should be impossible to factor...
            prime_factors = p.factor_left_detail()

            assert prime_factors.content == 1
            assert abs(prime_factors.unit) == 1
            assert len(prime_factors.primes) == 1
            assert abs(prime_factors.primes[0]) == abs(p)

            q, r = rdivmod(p, prime_factors.primes[0])
            assert not r
            assert abs(q) == 1


class TestFactorLeft(HurwitzIntTests):
    """Tests for factor_left"""

    def test_main(self):
        """Validate factor_left returns factors whose product is the original number."""
        factors = self.b_int.factor_left()

        ans = prod_left(factors)

        self.assert_equal(self.b_int, ans)

    def test_examples(self):
        """Validate factor_left works as expected for some given examples."""
        for n in (
            hurwitzint(2, 3, 4, 53),
            hurwitzint(1, 1, 1, 6),
            hurwitzint(1, 1, 2, 15),
            hurwitzint(17 * 31, 0, 0, 0),
        ):
            factors = n.factor_left()

            ans = prod_left(factors)

            self.assert_equal(n, ans)

    def test_with_content_and_multiple_factors(self):
        """Validate factor_left does not apply scalar content more than once."""
        n = hurwitzint(6, 2, 4, 0)

        factors = n.factor_left(canonical=False)

        assert n.factor_left_detail(canonical=False).content > 1
        assert len(n.factor_left_detail(canonical=False).primes) > 1

        ans = prod_left(factors)

        self.assert_equal(n, ans)

    def test_canonical_with_content_and_multiple_factors(self):
        """Validate canonical factor_left works when scalar content is present."""
        n = hurwitzint(6, 2, 4, 0)

        factors = n.factor_left(canonical=True)

        assert n.factor_left_detail(canonical=True).content > 1
        assert len(n.factor_left_detail(canonical=True).primes) > 1

        ans = prod_left(factors)

        self.assert_equal(n, ans)


class TestRepr(HurwitzIntTests):
    """Validate the repr"""

    def test_repr(self):
        """Verify some basic examples"""

        assert repr(hurwitzint(1, 2, 3, 4)) == "(1+2i+3j+4k)"
        assert repr(hurwitzint(1, 3, 5, 7, half=True)) == "(1+3i+5j+7k)/2"
        assert repr(hurwitzint(0, 0, 0, 5)) == "5k"
        assert repr(hurwitzint(0, 0, 0, -5)) == "-5k"
        assert repr(hurwitzint(2, 0, 0, 0)) == "(2+0i+0j+0k)"
        assert repr(hurwitzint(1, 1, 1, 1)) == "(1+i+j+k)"
        assert repr(hurwitzint(-1, -1, -1, -1)) == "(-1-i-j-k)"
