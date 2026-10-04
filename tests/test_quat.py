from __future__ import annotations

import copy
import operator
import os
import pickle
import random

from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import product, starmap
from math import floor, gcd, isqrt
from pathlib import Path
from types import MappingProxyType

import pytest

from hurwitz import HurwitzQuaternion
from sympy import factorint, isprime

import quatint.quat

from quatint.quat import (
    NonCommutativeFactorization,
    gcd_left,
    gcd_right,
    hurwitzint,
    prod_left,
    prod_right,
    rdivmod,
    xgcd_left,
    xgcd_right,
)

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

    # Divisors for the division wide searches, with norms from 1 to 41, both kinds, and negative parts
    division_divisors = (
        hurwitzint(1, 1, 1, 1, half=True),  # norm 1, a unit, so every division is exact
        hurwitzint(1, 1, 0, 0),  # norm 2
        hurwitzint(1, -1, 1, 0),  # norm 3
        hurwitzint(3, -1, 1, 1, half=True),  # norm 3
        hurwitzint(2, 0, 0, 0),  # norm 4, an integer
        hurwitzint(1, 2, 0, 0),  # norm 5
        hurwitzint(-3, 3, -1, 1, half=True),  # norm 5
        hurwitzint(1, 2, -1, 0),  # norm 6
        hurwitzint(2, -1, 1, 1),  # norm 7
        hurwitzint(-5, 1, 3, -1, half=True),  # norm 9
        hurwitzint(1, 2, 3, 4),  # norm 30
        hurwitzint(3, -5, 7, 9, half=True),  # norm 41
    )

    def setup_method(self, _):
        """Setup some test data"""
        self.a = HurwitzQuaternion(1, 2, 3, 4)
        self.b = HurwitzQuaternion(2, 3, 4, 5)

        self.a_int = hurwitzint(1, 2, 3, 4)
        self.b_int = hurwitzint(2, 3, 4, 5)

    @staticmethod
    def assert_equal(res: tuple | list | HurwitzQuaternion | hurwitzint, res_int: hurwitzint):
        """Validate the hurwitzint is equal to the validation object, and that it is still backed by integers"""
        if isinstance(res, HurwitzQuaternion):
            res = [x * 2 for x in res]

        assert list(res) == list(res_int)

        assert isinstance(res_int.a, int)
        assert isinstance(res_int.b, int)
        assert isinstance(res_int.c, int)
        assert isinstance(res_int.d, int)

        assert isinstance(res_int, hurwitzint)

    @staticmethod
    def wide_search_values(bound: int = 3):
        """
        Every Lipschitz integer with components in [-bound, bound],
            and every half-integer with numerators in [1 - 2*bound, 2*bound - 1] (so [-5, 5] for the default bound)

        Yields:
            hurwitzint: Each of those values.
        """
        for a, b, c, d in product(range(-bound, bound + 1), repeat=4):
            yield hurwitzint(a, b, c, d)

        for a, b, c, d in product(range(1 - 2 * bound, 2 * bound, 2), repeat=4):
            yield hurwitzint(a, b, c, d, half=True)

    @staticmethod
    def rand_hurwitzint(rng: random.Random, bound: int) -> hurwitzint:
        """A random nonzero Hurwitz integer with parts in [-bound, bound], a half-integer half of the time"""
        while True:
            if rng.random() < 0.5:
                x = hurwitzint(*(rng.randint(-bound, bound) for _ in range(4)))
            else:
                x = hurwitzint(*(2 * rng.randint(-bound, bound - 1) + 1 for _ in range(4)), half=True)

            if x:
                return x

    @staticmethod
    def canonical_remainder(a: hurwitzint, b: hurwitzint, *, right: bool = False) -> hurwitzint:
        """
        Brute-force the remainder division should leave in a = q*b + r (or a = b*q + r): of every r a Hurwitz
            quotient q can leave, the one of least norm, and of those, the one with the largest numerator tuple.

        In numerator units the exact quotient is U/n, where n = N(b) and U = a*conj(b) (or conj(b)*a). A nearest
            Hurwitz integer has, in every part, one of the two even integers around U_i/n, or else one of the two odd
            ones around it, so trying all 2 * 2**4 of those is sure to include every remainder of least norm.

        Returns:
            hurwitzint: The remainder.
        """
        n = abs(b)
        U = list(b.conjugate() * a) if right else list(a * b.conjugate())

        remainders = []
        for parity in (0, 1):
            # The largest integer of this parity that is <= U_i/n, and so the one 2 above it is > U_i/n
            lows = [u // n - ((u // n - parity) & 1) for u in U]
            for s0, s1, s2, s3 in product((0, 2), repeat=4):
                q = hurwitzint(lows[0] + s0, lows[1] + s1, lows[2] + s2, lows[3] + s3, half=True)
                remainders.append(a - b * q if right else a - q * b)

        return min(remainders, key=lambda r: (abs(r), [-part for part in r]))


class TestEq(HurwitzIntTests):
    """Tests for __eq__"""

    def test_main(self):
        """Basic equals tests"""
        c = hurwitzint(1, 2, 3, 4)
        assert self.a_int == c
        assert self.b_int != c

    def test_every_part_counts(self):
        """Changing any one part gives a different value, so == has to compare all four parts"""
        for x in (hurwitzint(1, 2, 3, 4), hurwitzint(3, 5, 7, 9, half=True)):
            for idx in range(4):
                for step in (2, -2):  # a step of 2 keeps the parity, so this is still a Hurwitz integer
                    parts = list(x)
                    parts[idx] += step
                    y = hurwitzint(*parts, half=True)

                    assert x != y
                    assert y != x
                    assert not operator.eq(x, y)

    def test_other_types_are_never_equal(self):
        """A hurwitzint never equals something that is not a number, not even a sequence of its own parts"""
        x = hurwitzint(1, 2, 3, 4)
        for other in ("x", None, [2, 4, 6, 8], (2, 4, 6, 8), object()):
            assert x != other
            assert other != x
            assert not operator.eq(x, other)


class TestEqualityWithNumbers(HurwitzIntTests):
    """Tests for __eq__ and __hash__ against plain Python numbers"""

    def test_real_values_compare_and_hash_like_int(self):
        """A real hurwitzint compares and hashes exactly like the int it equals, against ints, floats and complexes"""
        # 2**53 + 1 and larger do not survive float(), and -1 is the one int whose hash is not itself
        ints = [0, 1, -1, 2, -4, 7, 2**53, 2**53 + 1, 2**64 + 1, -(2**70)]
        numbers = [*ints, *(float(n) for n in ints), *(complex(n, 0) for n in ints)]
        numbers += [1.5, -0.5, -0.0, 1.9, float("inf"), float("-inf"), float("nan"), 1j, complex(2, 1)]

        for n in ints:
            x = hurwitzint(n)
            for number in numbers:
                assert (x == number) == (n == number)
                assert (number == x) == (number == n)
                assert operator.ne(x, number) == operator.ne(n, number)
                assert operator.ne(number, x) == operator.ne(number, n)

                if x == number:
                    assert hash(x) == hash(number)

    def test_non_real_values_never_equal_numbers(self):
        """Anything with an i, j or k part (so every half-integer) never equals a Python number"""
        numbers = (0, 1, 2, 3, 0.5, 1.5, 3.0, 1j, 1 + 1j, complex(3, 1))
        for x in (hurwitzint(0, 1, 0, 0), hurwitzint(3, 0, 0, 1), hurwitzint(1, 0, 2, 0),
                  hurwitzint(1, 1, 1, 1, half=True), hurwitzint(3, -1, 1, -1, half=True)):
            for number in numbers:
                assert x != number
                assert number != x
                assert not operator.eq(x, number)

    def test_dict_and_set_membership(self):
        """A real hurwitzint and the number it equals are the same dict key and set member"""
        assert {1: "int"}[hurwitzint(1)] == "int"
        assert {hurwitzint(3): "hurwitzint"}[3.0] == "hurwitzint"
        assert hurwitzint(-1) in {-1, 5}

        equal_values = [hurwitzint(2), 2, 2.0, complex(2, 0)]
        assert len(set(equal_values)) == 1


class TestInit(HurwitzIntTests):
    """Tests for __init__"""

    def test_floats_are_truncated(self):
        """Float components are truncated with int(), the same in the compiled and pure-Python builds"""
        assert hurwitzint(1.9, -2.9, 3.0, 0.5) == hurwitzint(1, -2, 3, 0)
        assert hurwitzint(0.5, 0.5, 0.5, 0.5) == 0
        assert hurwitzint(3.0, -5.0, 7.9, 9.0, half=True) == hurwitzint(3, -5, 7, 9, half=True)

    def test_bools_become_ints(self):
        """Bools (an int subclass) are stored as plain ints, just like int() makes them"""
        half_unit = hurwitzint(True, True, True, True, half=True)  # ruff: ignore[boolean-positional-value-in-call]
        lipschitz = hurwitzint(True, False, True, False)  # ruff: ignore[boolean-positional-value-in-call]

        for x in (half_unit, lipschitz):
            assert all(type(n) is int for n in x)

        assert repr(half_unit) == "(1+i+j+k)/2"

    def test_big_ints_are_exact(self):
        """Components far past 2**53 come through exactly, so ints are never squeezed through a float"""
        for n in (2**53 + 1, 2**64 + 1, -(2**70) - 1, 10**30 + 7):
            assert list(hurwitzint(n, -n, n + 2, 3)) == [2 * n, -2 * n, 2 * n + 4, 6]
            assert list(hurwitzint(2 * n + 1, 1, -1, 3, half=True)) == [2 * n + 1, 1, -1, 3]

    def test_parity_is_checked(self):
        """Numerators given with half=True must be all even or all odd, since anything else is not a Hurwitz integer"""
        for parts in ((1, 2, 3, 4), (1, 1, 1, 2), (1, 0, 0, 0), (2, 3, 2, 2)):
            with pytest.raises(ValueError, match="same parity"):
                hurwitzint(*parts, half=True)

        # Without half=True the parts are whole numbers, so any four of them make a Hurwitz integer
        assert list(hurwitzint(1, 0, 0, 0)) == [2, 0, 0, 0]

    def test_missing_parts_are_zero(self):
        """Any parts left out are zero"""
        assert hurwitzint() == 0
        assert list(hurwitzint()) == [0, 0, 0, 0]
        assert hurwitzint(5) == 5
        assert list(hurwitzint(5, -6)) == [10, -12, 0, 0]

    def test_only_numbers(self):
        """A part that isn't an int or float raises TypeError in both builds, even ones int() would take"""
        for bad in ("1", Fraction(7, 2), Decimal(3), hurwitzint(5), complex(1, 0), None):
            for parts in ((bad,), (1, bad), (1, 1, 1, bad)):
                with pytest.raises(TypeError):
                    hurwitzint(*parts)

                with pytest.raises(TypeError):
                    hurwitzint(*parts, half=True)


class TestImmutable(HurwitzIntTests):
    """Tests that a hurwitzint (or a factorization) cannot be changed once it is made"""

    def test_components_are_read_only(self):
        """Setting or deleting a, b, c or d raises AttributeError, and leaves the value (and its hash) alone"""
        x = hurwitzint(1, 2, 3, 4)
        values = {x: "x"}

        for name in ("a", "b", "c", "d"):
            with pytest.raises(AttributeError):
                setattr(x, name, 3)

            with pytest.raises(AttributeError):
                delattr(x, name)

        assert list(x) == [2, 4, 6, 8]
        assert values[hurwitzint(1, 2, 3, 4)] == "x"

    def test_factorization_is_frozen(self):
        """A factorization's fields are read-only too, and it has no __dict__ to take new ones"""
        factors = hurwitzint(2, 3, 4, 53).factor_right_detail()

        # Only the fields, since a name that isn't one raises TypeError instead in pure Python before 3.14
        #   (a CPython bug in the __setattr__ of a frozen dataclass with slots)
        for name in ("content", "unit", "primes", "direction"):
            with pytest.raises(AttributeError):  # FrozenInstanceError is an AttributeError
                setattr(factors, name, 1)

            with pytest.raises(AttributeError):
                delattr(factors, name)

        assert not hasattr(factors, "__dict__")
        assert factors == hurwitzint(2, 3, 4, 53).factor_right_detail()


class TestPickle(HurwitzIntTests):
    """Tests for pickling and copying, which is what multiprocessing (and copy.deepcopy) rely on"""

    @staticmethod
    def round_trips(obj: object) -> list:
        """Return obj after a pickle round trip at every protocol, a deepcopy, and a shallow copy"""
        out = [pickle.loads(pickle.dumps(obj, protocol)) for protocol in range(pickle.HIGHEST_PROTOCOL + 1)]
        out.extend((copy.deepcopy(obj), copy.copy(obj)))
        return out

    def test_hurwitzint(self):
        """A hurwitzint comes back equal, and hashing the same, from every pickle protocol and through copy"""
        for x in (hurwitzint(1, 2, 3, 4), hurwitzint(3, -5, 7, 9, half=True), hurwitzint(0), hurwitzint(-7),
                  hurwitzint(10**30, -1, 0, 2**70)):
            for y in self.round_trips(x):
                assert type(y) is hurwitzint
                assert y == x
                assert hash(y) == hash(x)

    def test_factorization(self):
        """A factorization comes back equal (content, unit, primes and direction), from every protocol and copy"""
        n = 30 * hurwitzint(3, 5, 7, 9, half=True)
        cases = ((n, n.factor_right_detail()), (n, n.factor_left_detail()),
                 (n, n.factor_left_detail().expand_content()), (hurwitzint(0), hurwitzint(0).factor_right_detail()))
        for value, factors in cases:
            for copied in self.round_trips(factors):
                assert type(copied) is NonCommutativeFactorization
                assert copied == factors
                assert hash(copied) == hash(factors)
                assert copied.prod() == value


class TestConversions(HurwitzIntTests):
    """Tests for int(), float(), complex() and __index__ of a hurwitzint"""

    def test_real_values(self):
        """A real hurwitzint converts to the int, float and complex it equals, exactly as that int would"""
        index = hurwitzint.__index__  # Called directly, since the compiled build doesn't use it for operator.index yet
        for n in (0, 1, -1, 7, -12, 2**53 + 1, -(10**30) - 7):
            x = hurwitzint(n)
            for convert in (int, float, complex):
                result = convert(x)
                assert type(result) is type(convert(n))
                assert result == convert(n)

            assert type(index(x)) is int
            assert index(x) == n

    def test_non_real_values_raise(self):
        """Anything with an i, j or k part (so every half-integer) equals no Python number, so converting raises"""
        for x in (hurwitzint(0, 1, 0, 0), hurwitzint(3, 0, 0, -1), hurwitzint(1, 1, 1, 1, half=True)):
            for convert in (int, float, complex, hurwitzint.__index__):
                with pytest.raises(TypeError, match="not real"):
                    convert(x)

    def test_index_where_python_wants_an_int(self):
        """
        Through __index__, a real hurwitzint works where Python wants an int, like range() or a[x], in pure Python.

        The compiled build can't do that yet, since mypyc doesn't put __index__ in the type's slot (see the TODO on
            __index__). Once it does, the compiled half of this fails, as a reminder to drop the TODO and this branch.
        """
        uses = (operator.index, lambda n: list(range(n)), lambda n: "abcd"[n], lambda n: n * [1], lambda n: gcd(n, 12))
        expected = (3, [0, 1, 2], "d", [1, 1, 1], 3)
        if quatint.quat.__file__.endswith(".py"):
            for use, result in zip(uses, expected, strict=True):
                assert use(hurwitzint(3)) == result
        else:
            for use in uses:
                with pytest.raises(TypeError):
                    use(hurwitzint(3))

        # Anything that isn't real never works as an int
        for use in uses:
            with pytest.raises(TypeError):
                use(hurwitzint(3, 1, 0, 0))

    def test_too_big_for_a_float(self):
        """Like an int, a real hurwitzint too big for a float raises OverflowError"""
        with pytest.raises(OverflowError):
            float(hurwitzint(10**400))


class TestComponents(HurwitzIntTests):
    """Tests for reading a hurwitzint's parts: a, b, c and d, len, iteration, indexing, den and is_lipschitz"""

    def test_parts_are_numerators(self):
        """However they are read, the parts are the numerators over 2, and they rebuild the value with half=True"""
        for x, numerators in ((hurwitzint(1, -2, 3, 0), [2, -4, 6, 0]),
                              (hurwitzint(3, -5, 7, 9, half=True), [3, -5, 7, 9])):
            assert len(x) == 4
            assert list(x) == numerators
            assert [x[0], x[1], x[2], x[3]] == numerators
            assert [x[-4], x[-3], x[-2], x[-1]] == numerators  # Counting back from the end, as with a tuple
            assert [x.a, x.b, x.c, x.d] == numerators
            assert hurwitzint(*x, half=True) == x

    def test_index_out_of_range(self):
        """Indexing past the four parts, from either end, raises IndexError"""
        x = hurwitzint(1, 2, 3, 4)
        for idx in (4, 5, 100, -5, -6, -100):
            with pytest.raises(IndexError):
                operator.getitem(x, idx)

    def test_index_types(self):
        """Only an int indexes a hurwitzint: anything else, a slice included, raises TypeError in both builds"""
        x = hurwitzint(1, 2, 3, 4)
        for idx in (slice(1, None), slice(None), 1.0, "1", None):
            with pytest.raises(TypeError):
                operator.getitem(x, idx)

        assert x[True] == x[1]  # A bool is an int, as with a tuple

    def test_den_and_is_lipschitz(self):
        """Every part is over den == 2, and is_lipschitz is whether they all come out whole"""
        for x in (hurwitzint(1, -2, 3, 0), hurwitzint(0), hurwitzint(3, -5, 7, 9, half=True)):
            assert x.den == 2
            assert x.is_lipschitz == all(n % x.den == 0 for n in x)

        assert hurwitzint(1, 2, 3, 4).is_lipschitz
        assert not hurwitzint(1, 1, 1, 1, half=True).is_lipschitz


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
        """Test float * hurwitzint"""
        for i in range(100):
            res_int = float(i) * self.a_int

            self.assert_equal((2 * i, 4 * i, 6 * i, 8 * i), res_int)


class TestScalarOperands(HurwitzIntTests):
    """Tests for +, - and * with an int or float on either side"""

    def test_matches_hurwitzint(self):
        """
        A number on either side of +, - or * gives exactly what the hurwitzint it truncates to gives,
            for negative, big and fractional numbers and bools, with Lipschitz and half-integer values
        """
        numbers = (0, 1, -1, 7, -12, 2**30, -(2**31) - 5, 2**62 + 3, -(10**30) - 7, True, False,
                   2.9, -2.9, 0.5, -0.0, 1e20)
        values = (hurwitzint(1, -2, 3, 0), hurwitzint(3, -5, 7, 9, half=True), hurwitzint(0),
                  hurwitzint(-(10**20), 1, -1, 3), hurwitzint(2**31 + 1, -(2**31) - 3, 5, -7, half=True))
        for x in values:
            for n in numbers:
                h = hurwitzint(n)
                for result, expected in ((x + n, x + h), (n + x, h + x), (x - n, x - h), (n - x, h - x),
                                         (x * n, x * h), (n * x, h * x)):
                    assert result == expected
                    assert type(result) is hurwitzint
                    assert all(type(part) is int for part in result)


class TestUnsupportedOperands(HurwitzIntTests):
    """Tests for +, - and * with operand types hurwitzint does not support"""

    def test_raise_type_error(self):
        """An unsupported operand raises TypeError, on either side of +, - or *, as it would with an int"""
        for op in (operator.add, operator.sub, operator.mul):
            for other in (None, "a", [1]):
                with pytest.raises(TypeError):
                    op(self.a_int, other)

                with pytest.raises(TypeError):
                    op(other, self.a_int)

    def test_defer_to_reflected_ops(self):
        """An operand type hurwitzint does not support gets to try its own reflected method"""

        class Other:
            def __radd__(self, _: object) -> str:
                return "radd"

            def __rsub__(self, _: object) -> str:
                return "rsub"

            def __rmul__(self, _: object) -> str:
                return "rmul"

        other = Other()
        assert self.a_int + other == "radd"
        assert self.a_int - other == "rsub"
        assert self.a_int * other == "rmul"


class TestPow(HurwitzIntTests):
    """Tests for __pow__"""

    def test_pow(self):
        """Powers match repeated multiplication, and stay exact for big results"""
        x = hurwitzint(1, 2, 3, 4)
        assert x ** 0 == 1
        assert x ** 1 == x
        assert x ** 3 == x * x * x

        # (1+i)**2 == 2i, so (1+i)**100 == (2i)**50 == -2**50
        assert hurwitzint(1, 1, 0, 0) ** 100 == -(2**50)

    def test_float_exponent_is_truncated(self):
        """A float exponent is truncated with int(), like any other float meeting a hurwitzint"""
        x = hurwitzint(1, 2, 3, 4)
        assert pow(x, 2.0) == x * x
        assert pow(x, 2.9) == x * x
        assert pow(x, 0.5) == 1

    def test_non_number_exponents(self):
        """A non-number exponent raises TypeError in both builds, and a type with its own __rpow__ gets to handle it"""
        x = hurwitzint(1, 2, 3, 4)
        for exp in ("2", "a", None, [2]):
            with pytest.raises(TypeError):
                pow(x, exp)

        class Other:
            def __rpow__(self, _: object) -> str:
                return "rpow"

        assert pow(x, Other()) == "rpow"

    def test_hurwitzint_exponents(self):
        """
        Nothing takes a hurwitzint as an exponent, a real one included, so whatever the base, ** and pow raise
            TypeError in both builds (the compiled one used to recurse until RecursionError, see the TODO on __rpow__)
        """
        x = hurwitzint(1, 2, 3, 4)
        for exp in (x, hurwitzint(2), hurwitzint(1, 1, 1, 1, half=True)):
            for base in (x, hurwitzint(2), 2, 2.5, None, "a"):
                with pytest.raises(TypeError):
                    operator.pow(base, exp)

                with pytest.raises(TypeError):
                    pow(base, exp, 5)

            with pytest.raises(TypeError):
                operator.ipow(x, exp)

    def test_power_laws(self):
        """x**(m+n) == x**m * x**n and (x**m)**n == x**(m*n) for random x, and norms and conjugates follow along"""
        rng = random.Random(6_000)
        for bound in (3, 10**4, 10**12):
            for _ in range(30):
                x = self.rand_hurwitzint(rng, bound)
                m, n = rng.randint(0, 8), rng.randint(0, 8)

                assert x ** (m + n) == x ** m * x ** n
                assert (x ** m) ** n == x ** (m * n)
                assert abs(x ** n) == abs(x) ** n
                assert x.conjugate() ** n == (x ** n).conjugate()

    def test_unit_orders(self):
        """Each unit's order is 1, 2, 3, 4 or 6, just like in the binary tetrahedral group the 24 of them form"""
        orders = []
        for u in hurwitzint.UNITS:
            powers = [u ** n for n in range(1, 13)]
            orders.append(1 + powers.index(1))

            assert powers[-1] == 1  # u**12 == 1, since 12 is a multiple of every order

        # 1, then -1, then the 8 like (-1+i+j+k)/2, the 6 like i, and the 8 like (1+i+j+k)/2
        assert sorted(orders) == [1, 2] + [3] * 8 + [4] * 6 + [6] * 8

    def test_negative_powers_of_non_units_raise(self):
        """A non-unit has no inverse among the Hurwitz integers, so a negative power of one raises ValueError"""
        for x in (hurwitzint(0), hurwitzint(2), hurwitzint(1, 1, 0, 0), hurwitzint(3, 5, 7, 9, half=True)):
            for e in (-1, -2, -1.5):
                with pytest.raises(ValueError, match="Negative powers"):
                    pow(x, e)

    def test_negative_powers_of_units(self):
        """A unit's negative powers are the powers of its inverse, so they multiply back to 1 with its positive ones"""
        one = hurwitzint(1)
        for unit in hurwitzint.UNITS:
            for n in range(1, 13):
                assert unit ** -n == (~unit) ** n
                assert unit ** -n * unit ** n == unit ** n * unit ** -n == one

        i = hurwitzint(0, 1, 0, 0)
        assert i ** -1 == -i
        assert i ** -2 == -1
        assert pow(i, -1.9) == pow(i, -1)  # A float exponent is truncated with int(), toward 0


class TestAlgebra(HurwitzIntTests):
    """Tests that the arithmetic follows the laws of quaternion algebra, for random values of every size"""

    @classmethod
    def random_triples(cls, seed: int):
        """Seeded random triples of Hurwitz integers, with parts from 3 up to around 10**30"""
        rng = random.Random(seed)
        for bound in (3, 10**4, 10**30):
            for _ in range(100):
                yield cls.rand_hurwitzint(rng, bound), cls.rand_hurwitzint(rng, bound), cls.rand_hurwitzint(rng, bound)

    def test_multiplication_table(self):
        """i, j and k multiply as i^2 = j^2 = k^2 = ijk = -1 says, which pins down every sign in the product"""
        i, j, k = hurwitzint(0, 1, 0, 0), hurwitzint(0, 0, 1, 0), hurwitzint(0, 0, 0, 1)

        assert i * i == j * j == k * k == i * j * k == -1
        assert (i * j, j * k, k * i) == (k, i, j)
        assert (j * i, k * j, i * k) == (-k, -i, -j)

    def test_ring_laws(self):
        """Addition and multiplication are associative, addition is commutative, and multiplication distributes"""
        for a, b, c in self.random_triples(5_000):
            assert (a + b) + c == a + (b + c)
            assert a + b == b + a
            assert (a * b) * c == a * (b * c)
            assert a * (b + c) == a * b + a * c
            assert (a + b) * c == a * c + b * c

    def test_identities_and_negation(self):
        """0 and 1 are the identities, a - b is a + (-b), and negation undoes itself"""
        for a, b, _ in self.random_triples(5_001):
            assert a + 0 == 0 + a == a
            assert a * 1 == 1 * a == a
            assert a * 0 == 0 * a == 0
            assert a - b == a + (-b) == -(b - a)
            assert a - a == 0
            assert a * -1 == -a

            negated = -a
            assert -negated == +a == a

    def test_integers_commute(self):
        """An integer commutes with everything, even though Hurwitz integers generally do not commute"""
        commuting_pairs = 0
        for a, b, _ in self.random_triples(5_002):
            for m in (2, -3, 10**20):
                assert m * a == a * m
                assert hurwitzint(m) * a == a * hurwitzint(m)

            commuting_pairs += a * b == b * a

        assert commuting_pairs < 30

    def test_norm_is_multiplicative(self):
        """N(ab) = N(a)N(b), and the norm of anything nonzero is positive"""
        for a, b, _ in self.random_triples(5_003):
            assert abs(a * b) == abs(a) * abs(b)
            assert abs(a) > 0

        assert abs(hurwitzint(0)) == 0

    def test_conjugate(self):
        """Conjugation undoes itself and reverses products, and a times its conjugate is its norm"""
        for a, b, _ in self.random_triples(5_004):
            assert a.conjugate().conjugate() == a
            assert (a * b).conjugate() == b.conjugate() * a.conjugate()
            assert (a + b).conjugate() == a.conjugate() + b.conjugate()
            assert a * a.conjugate() == a.conjugate() * a == abs(a)
            assert abs(a.conjugate()) == abs(a)

            # a + conj(a) is twice the real part, and the real part is a.a / 2
            assert a + a.conjugate() == a.a


class TestTrace(HurwitzIntTests):
    """Tests for trace, x + conj(x) as an int"""

    def test_examples(self):
        """Some traces worked out by hand: twice the real part, which is a whole number for half-integers too"""
        examples = (
            (hurwitzint(1, 2, 3, 4), 2),
            (hurwitzint(3, -5, 7, 9, half=True), 3),
            (hurwitzint(-1, 1, 1, 1, half=True), -1),
            (hurwitzint(0, 1, 0, 0), 0),
            (hurwitzint(-7), -14),
            (hurwitzint(0), 0),
            (hurwitzint(10**30, -1, 0, 2), 2 * 10**30),
        )
        for x, expected in examples:
            assert x.trace == expected
            assert type(x.trace) is int
            assert x + x.conjugate() == expected

    def test_trace_and_norm_give_the_polynomial(self):
        """
        Every x is a root of x**2 - trace*x + norm, and traces add like the values do. The traces of x*y and y*x are the
            same too, though x*y and y*x usually differ
        """
        rng = random.Random(19_000)
        for bound in (3, 10**4, 10**30):
            for _ in range(100):
                x, y = self.rand_hurwitzint(rng, bound), self.rand_hurwitzint(rng, bound)

                assert x * x - x.trace * x + abs(x) == 0
                assert (x + y).trace == x.trace + y.trace
                assert (5 * x).trace == 5 * x.trace
                assert x.conjugate().trace == x.trace
                assert (x * y).trace == (y * x).trace


class TestDiv(HurwitzIntTests):
    """Tests for __truediv__ and __floordiv__"""

    def test_div(self):
        """Test hurwitzint / hurwitzint"""
        res_q, res_r = self.a.euclidean_division(self.b)
        res_int_q, res_int_r = divmod(self.a_int, self.b_int)

        self.assert_equal(res_q, res_int_q)
        self.assert_equal(res_r, res_int_r)

    def test_wide_search(self):
        """
        Divide every small dividend by a spread of divisors, and check each remainder is as small as promised.

        a == q*b + r alone holds for any q at all, since divmod works r out as a - q*b. A correct quotient is what
            makes the remainder small: every quaternion is within norm 1/2 of a Hurwitz integer, so 2*N(r) <= N(b).
        """
        for b in self.division_divisors:
            for a in self.wide_search_values():
                q, r = divmod(a, b)

                assert q * b + r == a
                assert 2 * abs(r) <= abs(b)

    def test_random_remainder_is_canonical(self):
        """
        For random pairs of every size, the remainder is as small as any Hurwitz quotient could make it, and of those
            that small, the one with the largest numerator tuple
        """
        rng = random.Random(1_000)
        for bound_a, bound_b in ((10, 3), (10, 10**4), (10**4, 10**2), (10**12, 10**5), (10**30, 10**12)):
            for _ in range(150):
                a = self.rand_hurwitzint(rng, bound_a)
                b = self.rand_hurwitzint(rng, bound_b)
                q, r = divmod(a, b)

                assert q * b + r == a
                assert 2 * abs(r) <= abs(b)
                assert r == self.canonical_remainder(a, b)

    def test_small_remainders_are_canonical(self):
        """Among small values, where several quotients often leave the least norm, the remainder is still canonical"""
        rng = random.Random(1_001)
        values = list(self.wide_search_values(2))
        for b in self.division_divisors:
            for a in rng.sample(values, 60):
                assert a % b == self.canonical_remainder(a, b)

    def test_integer_divisors(self):
        """
        Dividing by an integer (an int, or a real hurwitzint) takes a shortcut, which still leaves the brute-forced
            remainder, and the same quotient and remainder from either side, for small values (where ties are common)
            and random ones of every size
        """
        rng = random.Random(1_002)
        small = rng.sample(list(self.wide_search_values(2)), 100)
        cases = [(a, m) for a in small for m in (1, 2, -2, 3, 4, -6, 7)]
        for bound in (10**4, 10**30):
            for _ in range(40):
                a = self.rand_hurwitzint(rng, bound)
                cases.extend((a, m) for m in (-1, 5, 12, -97, 10**9 + 7, -(2**64)))

        for a, m in cases:
            q, r = divmod(a, m)
            assert q * m + r == a
            assert r == self.canonical_remainder(a, hurwitzint(m))
            assert divmod(a, hurwitzint(m)) == a.rdivmod(m) == a.rdivmod(hurwitzint(m)) == (q, r)

    def test_remainder_depends_only_on_class(self):
        """
        Values congruent modulo the left multiples of b leave the same remainder, and so do the divisors with the same
            left multiples (u*b for a unit u), even where several quotients tie, as they often do for small values
        """
        shifts = (hurwitzint(1), hurwitzint(0, -1, 0, 0), hurwitzint(1, -1, 1, 1, half=True))
        j = hurwitzint(0, 0, 1, 0)
        for b in self.division_divisors:
            for a in self.wide_search_values(2):
                r = a % b
                for h in shifts:
                    assert (a + h * b) % b == r

                assert a % (j * b) == r

    def test_exact_multiples_divide_exactly(self):
        """Dividing q*b by b gives back exactly q with no remainder, for random q and b of every size"""
        rng = random.Random(2_000)
        for bound in (3, 10**4, 10**12, 10**30):
            for _ in range(250):
                q = self.rand_hurwitzint(rng, bound)
                b = self.rand_hurwitzint(rng, bound)

                assert divmod(q * b, b) == (q, 0)

    def test_divide_by_units(self):
        """Dividing by a unit u is exact: the quotient is a * u^-1, and nothing is left over"""
        for u in hurwitzint.UNITS:
            for a in self.division_divisors:
                assert divmod(a, u) == (a * u.inverse(), 0)

    def test_divmod_int_reversed(self):
        """Test divmod(int, hurwitzint), int // hurwitzint and int % hurwitzint"""
        for b in (self.b_int, hurwitzint(3, 5, 7, 9, half=True)):
            for i in range(-60, 61):
                q, r = divmod(i, b)
                res_q, res_r = divmod(hurwitzint(i), b)

                self.assert_equal(res_q, q)
                self.assert_equal(res_r, r)
                self.assert_equal(q, i // b)
                self.assert_equal(r, i % b)

                assert q * b + r == hurwitzint(i)
                assert 2 * abs(r) <= abs(b)

    def test_divmod_float_reversed(self):
        """Test divmod(float, hurwitzint), float // hurwitzint and float % hurwitzint"""
        for i in range(-60, 61):
            q, r = divmod(float(i), self.b_int)
            res_q, res_r = divmod(hurwitzint(i), self.b_int)

            self.assert_equal(res_q, q)
            self.assert_equal(res_r, r)
            self.assert_equal(q, float(i) // self.b_int)
            self.assert_equal(r, float(i) % self.b_int)

    def test_int_and_float_divisors(self):
        """An int or float divisor divides like the hurwitzint it equals, and the same on either side"""
        for a in (self.a_int, hurwitzint(3, -5, 7, 9, half=True), hurwitzint(-17, 4, 0, 23)):
            for m in (1, -1, 2, 3, -3, 7, 3.0, -2.0):
                expected = divmod(a, hurwitzint(m))

                assert divmod(a, m) == expected
                assert a // m == expected[0]
                assert a / m == expected[0]
                assert a % m == expected[1]

                # An integer commutes with everything, so right-division by one is the same
                assert rdivmod(a, m) == expected

    def test_truediv_is_floordiv(self):
        """/ gives the same Euclidean quotient as //, with a hurwitzint, int or float on the left"""
        for b in self.division_divisors:
            for a in (self.a_int, hurwitzint(3, -5, 7, 9, half=True), hurwitzint(-17, 4, 0, 23)):
                assert a / b == a // b

            for i in (-7, 0, 5, 12):
                self.assert_equal(hurwitzint(i) // b, i / b)
                self.assert_equal(hurwitzint(i) // b, float(i) / b)

    def test_divide_by_zero(self):
        """Dividing by a zero hurwitzint, int or float raises ZeroDivisionError, with a hurwitzint on either side"""
        for op in (divmod, operator.floordiv, operator.mod, operator.truediv):
            for zero in (hurwitzint(0), 0, 0.0):
                with pytest.raises(ZeroDivisionError):
                    op(self.a_int, zero)

            with pytest.raises(ZeroDivisionError):
                op(5, hurwitzint(0))

    def test_unsupported_reversed_types_raise_type_error(self):
        """An unsupported left operand should raise TypeError, like it does with int"""
        for op in (divmod, operator.floordiv, operator.mod, operator.truediv):
            with pytest.raises(TypeError):
                op(None, self.b_int)

    def test_unsupported_types_raise_type_error(self):
        """An unsupported operand type should raise TypeError (as it does for int), not NotImplementedError"""
        for op in (divmod, operator.floordiv, operator.mod, operator.truediv):
            with pytest.raises(TypeError):
                op(self.a_int, "a")

    def test_unsupported_types_defer_to_reflected_ops(self):
        """An operand type hurwitzint does not support should get to try its own reflected method"""

        class Other:
            def __rdivmod__(self, _: object) -> str:
                return "rdivmod"

            def __rfloordiv__(self, _: object) -> str:
                return "rfloordiv"

            def __rmod__(self, _: object) -> str:
                return "rmod"

            def __rtruediv__(self, _: object) -> str:
                return "rtruediv"

        other = Other()
        assert divmod(self.a_int, other) == "rdivmod"
        assert self.a_int // other == "rfloordiv"
        assert self.a_int % other == "rmod"
        assert self.a_int / other == "rtruediv"


class TestRDiv(HurwitzIntTests):
    """Tests for rtruediv and rfloordiv"""

    def test_rdiv(self):
        r"""Test hurwitzint \ hurwitzint"""
        g = hurwitzint(1, 0, 0, 1)
        i = hurwitzint(0, 1, 0, 0)

        a = i * g

        q, r = a.rdivmod(g)
        assert not r
        assert g * q == a

    def test_wide_search(self):
        """The right-division version of TestDiv.test_wide_search: a == b*q + r, with 2*N(r) <= N(b)"""
        for b in self.division_divisors:
            for a in self.wide_search_values():
                q, r = rdivmod(a, b)

                assert b * q + r == a
                assert 2 * abs(r) <= abs(b)

    def test_random_remainder_is_canonical(self):
        """The right-division version of TestDiv.test_random_remainder_is_canonical"""
        rng = random.Random(3_000)
        for bound_a, bound_b in ((10, 3), (10, 10**4), (10**4, 10**2), (10**12, 10**5), (10**30, 10**12)):
            for _ in range(150):
                a = self.rand_hurwitzint(rng, bound_a)
                b = self.rand_hurwitzint(rng, bound_b)
                q, r = rdivmod(a, b)

                assert b * q + r == a
                assert 2 * abs(r) <= abs(b)
                assert r == self.canonical_remainder(a, b, right=True)

    def test_small_remainders_are_canonical(self):
        """The right-division version of TestDiv.test_small_remainders_are_canonical"""
        rng = random.Random(3_001)
        values = list(self.wide_search_values(2))
        for b in self.division_divisors:
            for a in rng.sample(values, 60):
                assert a.rmod(b) == self.canonical_remainder(a, b, right=True)

    def test_remainder_depends_only_on_class(self):
        """
        The right-division version of TestDiv.test_remainder_depends_only_on_class: values congruent modulo the right
            multiples of b, and divisors with the same right multiples (b*u for a unit u), leave the same remainder
        """
        shifts = (hurwitzint(1), hurwitzint(0, -1, 0, 0), hurwitzint(1, -1, 1, 1, half=True))
        j = hurwitzint(0, 0, 1, 0)
        for b in self.division_divisors:
            for a in self.wide_search_values(2):
                r = a.rmod(b)
                for h in shifts:
                    assert (a + b * h).rmod(b) == r

                assert a.rmod(b * j) == r

    def test_exact_multiples_divide_exactly(self):
        """Right-dividing b*q by b gives back exactly q with no remainder, for random q and b of every size"""
        rng = random.Random(4_000)
        for bound in (3, 10**4, 10**12, 10**30):
            for _ in range(250):
                q = self.rand_hurwitzint(rng, bound)
                b = self.rand_hurwitzint(rng, bound)

                assert rdivmod(b * q, b) == (q, 0)

    def test_divide_by_units(self):
        """Right-dividing by a unit u is exact: the quotient is u^-1 * a, and nothing is left over"""
        for u in hurwitzint.UNITS:
            for a in self.division_divisors:
                assert rdivmod(a, u) == (u.inverse() * a, 0)

    def test_right_helpers_match_rdivmod(self):
        """The rfloordiv and rtruediv methods give the quotient from rdivmod, and rmod gives its remainder"""
        for b in self.division_divisors:
            for a in (self.a_int, hurwitzint(3, -5, 7, 9, half=True), hurwitzint(-17, 4, 0, 23)):
                q, r = a.rdivmod(b)

                assert a.rfloordiv(b) == q
                assert a.rtruediv(b) == q
                assert a.rmod(b) == r

    def test_divide_by_zero(self):
        """Right-dividing by zero (a hurwitzint, int or float) raises ZeroDivisionError from every method"""
        x = self.a_int
        for method in (x.rdivmod, x.rfloordiv, x.rmod, x.rtruediv):
            for zero in (hurwitzint(0), 0, 0.0):
                with pytest.raises(ZeroDivisionError):
                    method(zero)

    def test_unsupported_types_raise_type_error(self):
        """An unsupported operand type should raise TypeError, not NotImplementedError"""
        x = self.a_int
        for method in (x.rdivmod, x.rfloordiv, x.rmod, x.rtruediv):
            with pytest.raises(TypeError):
                method("a")


class TestExactDiv(HurwitzIntTests):
    """Tests for exact_div_right and exact_div_left"""

    def test_examples(self):
        """Some quotients worked out by hand"""
        # y = 1+i+j has norm 3. It divides i*y on the right, but not on the left: y^-1 * i*y would be (i+2j+2k)/3
        y = hurwitzint(1, 1, 1, 0)
        x = hurwitzint(0, 1, 0, 0) * y
        assert x == hurwitzint(-1, 1, 0, 1)
        assert x.exact_div_right(y) == hurwitzint(0, 1, 0, 0)
        assert x.exact_div_left(y) is None

        # The quotient can be a true half-integer, like (1+i+j+k) / 2
        assert hurwitzint(1, 1, 1, 1).exact_div_right(2) == hurwitzint(1, 1, 1, 1, half=True)
        assert hurwitzint(1, 1, 1, 1).exact_div_left(2) == hurwitzint(1, 1, 1, 1, half=True)

        # 2 / (1+i) is 1-i, but 1 / (1+i) is (1-i)/2: numerators 1, -1, 0, 0 of mixed parity, so no Hurwitz integer
        assert hurwitzint(2).exact_div_right(hurwitzint(1, 1, 0, 0)) == hurwitzint(1, -1, 0, 0)
        assert hurwitzint(2).exact_div_left(hurwitzint(1, 1, 0, 0)) == hurwitzint(1, -1, 0, 0)
        assert hurwitzint(1).exact_div_right(hurwitzint(1, 1, 0, 0)) is None
        assert hurwitzint(1).exact_div_left(hurwitzint(1, 1, 0, 0)) is None

    def test_matches_divmod(self):
        """
        For every small pair, exact_div_right gives the quotient of divmod when its remainder is 0, and None when
            it is not, and exact_div_left does the same for rdivmod
        """
        for y in self.division_divisors:
            for x in self.wide_search_values(2):
                q, r = divmod(x, y)
                if r:
                    assert x.exact_div_right(y) is None
                else:
                    assert x.exact_div_right(y) == q

                q, r = rdivmod(x, y)
                if r:
                    assert x.exact_div_left(y) is None
                else:
                    assert x.exact_div_left(y) == q

    def test_exact_multiples(self):
        """
        q*y divides by y on the right and y*q on the left, back to exactly q, for random q and y of every size.
            Adding 1 leaves neither one a multiple of y, unless y is a unit (y*w == 1 + y*q would make 1 a multiple)
        """
        rng = random.Random(9_000)
        for bound in (3, 10**4, 10**12, 10**30):
            for _ in range(250):
                q = self.rand_hurwitzint(rng, bound)
                y = self.rand_hurwitzint(rng, bound)

                assert (q * y).exact_div_right(y) == q
                assert (y * q).exact_div_left(y) == q

                if not y.is_unit:
                    assert (q * y + 1).exact_div_right(y) is None
                    assert (y * q + 1).exact_div_left(y) is None

    def test_divide_by_units(self):
        """Every unit u divides everything exactly: the quotients are a * u^-1 on the right, and u^-1 * a on the left"""
        for u in hurwitzint.UNITS:
            for a in self.division_divisors:
                assert a.exact_div_right(u) == a * u.inverse()
                assert a.exact_div_left(u) == u.inverse() * a

    def test_int_and_float_divisors(self):
        """An int or float divisor divides like the hurwitzint it equals, and the same on either side"""
        for a in (hurwitzint(6, -4, 2, 0), hurwitzint(3, 3, 3, 3), hurwitzint(3, -5, 7, 9, half=True)):
            for m in (1, -1, 2, 3, -3, 3.0, -2.0, 2.9):
                expected = a.exact_div_right(hurwitzint(m))

                assert a.exact_div_right(m) == expected
                assert a.exact_div_left(m) == expected

        assert hurwitzint(6, -4, 2, 0).exact_div_right(2) == hurwitzint(3, -2, 1, 0)
        assert hurwitzint(6, -4, 2, 0).exact_div_right(4) is None

    def test_zero(self):
        """0 divides exactly into 0 by anything, and dividing by 0 raises ZeroDivisionError"""
        for y in (self.a_int, hurwitzint(3, -5, 7, 9, half=True), 7):
            assert hurwitzint(0).exact_div_right(y) == 0
            assert hurwitzint(0).exact_div_left(y) == 0

        x = self.a_int
        for method in (x.exact_div_right, x.exact_div_left):
            for zero in (hurwitzint(0), 0, 0.0):
                with pytest.raises(ZeroDivisionError):
                    method(zero)

    def test_unsupported_types_raise_type_error(self):
        """Exact division by something that is not a number raises TypeError, on either side"""
        x = self.a_int
        for method in (x.exact_div_right, x.exact_div_left):
            for other in ("a", None, [1]):
                with pytest.raises(TypeError):
                    method(other)


class TestDivides(HurwitzIntTests):
    """Tests for divides_right and divides_left"""

    def test_examples(self):
        """Some divisors worked out by hand"""
        # 1+i+j divides i*(1+i+j) on the right, but not on the left (see TestExactDiv.test_examples)
        y = hurwitzint(1, 1, 1, 0)
        x = hurwitzint(0, 1, 0, 0) * y
        assert y.divides_right(x) is True
        assert y.divides_left(x) is False

        # 1+i divides 2 == (1-i) * (1+i) == (1+i) * (1-i) on both sides, but not 1, which would leave (1-i)/2
        assert hurwitzint(1, 1, 0, 0).divides_right(2) is True
        assert hurwitzint(1, 1, 0, 0).divides_left(2) is True
        assert hurwitzint(1, 1, 0, 0).divides_right(1) is False
        assert hurwitzint(1, 1, 0, 0).divides_left(1) is False

        # A Hurwitz integer divides its norm on both sides, since N(y) == conj(y) * y == y * conj(y)
        y = hurwitzint(3, -5, 7, 9, half=True)
        assert y.divides_right(41) is True
        assert y.divides_left(41) is True

        # 2 divides 1+i+j+k == 2 * (1+i+j+k)/2, but not 1+i, which would leave the numerators 1, 1, 0, 0
        assert hurwitzint(2).divides_right(hurwitzint(1, 1, 1, 1)) is True
        assert hurwitzint(2).divides_left(hurwitzint(1, 1, 1, 1)) is True
        assert hurwitzint(2).divides_right(hurwitzint(1, 1, 0, 0)) is False
        assert hurwitzint(2).divides_left(hurwitzint(1, 1, 0, 0)) is False

    def test_matches_exact_div(self):
        """For every small pair, y divides x on a side exactly when x divides exactly by y on that side"""
        for y in self.division_divisors:
            for x in self.wide_search_values(2):
                assert y.divides_right(x) is (x.exact_div_right(y) is not None)
                assert y.divides_left(x) is (x.exact_div_left(y) is not None)

    def test_multiples(self):
        """
        For random q and y of every size, y divides q*y on the right and y*q on the left. Adding 1 leaves neither
            one a multiple of y, unless y is a unit (y*w == 1 + y*q would make 1 a multiple of y)
        """
        rng = random.Random(10_000)
        for bound in (3, 10**4, 10**12, 10**30):
            for _ in range(250):
                q = self.rand_hurwitzint(rng, bound)
                y = self.rand_hurwitzint(rng, bound)

                assert y.divides_right(q * y) is True
                assert y.divides_left(y * q) is True

                if not y.is_unit:
                    assert y.divides_right(q * y + 1) is False
                    assert y.divides_left(y * q + 1) is False

    def test_gcd_divides_both(self):
        """A right gcd divides both its arguments on the right, and a left gcd divides both on the left"""
        rng = random.Random(11_000)
        for bound in (3, 10**4, 10**12):
            for _ in range(100):
                a = self.rand_hurwitzint(rng, bound)
                b = self.rand_hurwitzint(rng, bound)

                g = a.gcd_right(b)
                assert g.divides_right(a) is True
                assert g.divides_right(b) is True

                g = a.gcd_left(b)
                assert g.divides_left(a) is True
                assert g.divides_left(b) is True

    def test_units_divide_everything(self):
        """Every unit divides everything, on either side"""
        for u in hurwitzint.UNITS:
            for x in self.division_divisors:
                assert u.divides_right(x) is True
                assert u.divides_left(x) is True

    def test_zero(self):
        """Everything divides 0, and 0 divides only 0, with no ZeroDivisionError either way"""
        for y in (self.a_int, hurwitzint(1), hurwitzint(3, -5, 7, 9, half=True), hurwitzint(0)):
            for zero in (hurwitzint(0), 0, 0.0):
                assert y.divides_right(zero) is True
                assert y.divides_left(zero) is True

        for x in (self.a_int, hurwitzint(1), 5, -3.0):
            assert hurwitzint(0).divides_right(x) is False
            assert hurwitzint(0).divides_left(x) is False

    def test_int_and_float_arguments(self):
        """An int or float argument is divided like the hurwitzint it equals"""
        for y in (hurwitzint(1, 1, 0, 0), hurwitzint(3), hurwitzint(3, -5, 7, 9, half=True)):
            for m in (6, -6, 7, 41, 41.0, 6.5):
                assert y.divides_right(m) is y.divides_right(hurwitzint(m))
                assert y.divides_left(m) is y.divides_left(hurwitzint(m))

    def test_unsupported_types_raise_type_error(self):
        """Asking whether something that is not a number is divisible raises TypeError, on either side"""
        y = self.a_int
        for method in (y.divides_right, y.divides_left):
            for other in ("a", None, [1]):
                with pytest.raises(TypeError):
                    method(other)


class TestNearestByParity:
    """Tests for _nearest_by_parity, which every division starts from"""

    def test_matches_exact_rounding(self):
        """
        For small and big values, each parity's integer is the one nearest u/n, at the distance given (times n), the
            smaller one when two are as near, and a distance of n means exactly that tie
        """
        values = (*range(-60, 61), 10**30 + 5, -(10**30) - 5, 2**64 + 1, -(2**64) - 1, 2**64, -(2**64))
        for n in (*range(1, 13), 2**64):
            for u in values:
                t = Fraction(u, n)
                even, even_distance, odd, odd_distance = quatint.quat._nearest_by_parity(u, n)
                for q, distance, parity in ((even, even_distance, 0), (odd, odd_distance, 1)):
                    # The integers of this parity on either side of t, with the smaller one on a tie
                    below = floor(t) - (floor(t) - parity) % 2
                    nearest = min((below, below + 2), key=lambda c, t=t: (abs(t - c), c))

                    assert q == nearest
                    assert distance == abs(u - n * q)
                    assert (distance == n) == (t == below + 1)


class TestMulHelper:
    """Tests for _mul, the multiplication that the arithmetic's hot paths go through"""

    def test_matches_plain_multiplication(self):
        """It gives exactly x * y for every combination of signs, with sizes on both sides of mypyc's limits"""
        magnitudes = (0, 1, 2, 3, 2**30 - 1, 2**30, 2**31 + 5, 2**62 - 1, 2**62, 2**63 + 1, 10**30 + 7)
        values = [sign * m for m in magnitudes for sign in (1, -1)]
        for x in values:
            for y in values:
                assert quatint.quat._mul(x, y) == x * y


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

    def test_units_form_a_group(self):
        """UNITS holds every Hurwitz integer of norm 1, and they are closed under multiplication and inverses"""
        # Every part of something with norm 1 is within 1 of zero, so these are all the candidates
        near_zero = list(starmap(hurwitzint, product((-1, 0, 1), repeat=4)))
        near_zero += [hurwitzint(*p, half=True) for p in product((-1, 1), repeat=4)]
        units = set(hurwitzint.UNITS)

        assert {x for x in near_zero if abs(x) == 1} == units
        assert len(units) == 24
        for u in units:
            assert u.inverse() in units
            for v in units:
                assert u * v in units


class TestIsIrreducible(HurwitzIntTests):
    """Tests for is_irreducible, which is whether the norm is a rational prime"""

    def test_examples(self):
        """Some values worked out by hand"""
        assert not hurwitzint(0).is_irreducible
        for unit in hurwitzint.UNITS:
            assert not unit.is_irreducible

        # Norms 2, 3, 3, 7 and 41
        for x in (hurwitzint(1, 1, 0, 0), hurwitzint(1, 1, 1, 0), hurwitzint(3, 1, 1, 1, half=True),
                  hurwitzint(2, 1, 1, 1), hurwitzint(3, -5, 7, 9, half=True)):
            assert x.is_irreducible

        # A rational prime p is conj(P) * P for a Hurwitz prime P of norm p, and 1+2i+3j+4k has norm 30
        for x in (hurwitzint(2), hurwitzint(-3), hurwitzint(7), hurwitzint(1, 2, 3, 4)):
            assert not x.is_irreducible

    def test_matches_factorization(self):
        """Exactly the values whose factorization is a single Hurwitz prime are irreducible"""
        for x in self.wide_search_values():
            factors = x.factor_right()
            assert x.is_irreducible is (len(factors) == 1 and abs(x) > 1)

    def test_associates_and_products(self):
        """A unit on either side keeps a prime irreducible, and a product of two non-units is never irreducible"""
        rng = random.Random(15_000)
        for p in (2, 3, 5, 13, 10**9 + 7):
            prime = hurwitzint.prime_of_norm(p)
            for u in rng.sample(hurwitzint.UNITS, 6):
                assert (u * prime).is_irreducible
                assert (prime * u).is_irreducible

        for bound in (3, 10**6):
            for _ in range(50):
                a, b = self.rand_hurwitzint(rng, bound), self.rand_hurwitzint(rng, bound)
                if not a.is_unit and not b.is_unit:
                    assert not (a * b).is_irreducible


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
            with pytest.raises(ValueError, match="only Hurwitz units have inverses"):
                n.inverse()

    def test_invert_operator(self):
        """~u is the inverse of a unit, the same as u.inverse(), and ~ raises ValueError for anything else"""
        one = hurwitzint(1)
        for unit in hurwitzint.UNITS:
            inv = ~unit
            assert inv == unit.inverse()
            assert unit * inv == inv * unit == one
            assert ~inv == unit

        for n in (hurwitzint(0), hurwitzint(2), hurwitzint(1, 1, 0, 0), hurwitzint(3, 5, 7, 9, half=True)):
            with pytest.raises(ValueError, match="only Hurwitz units have inverses"):
                operator.invert(n)


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

    gcd_cofactors = (hurwitzint(1, 2, 3, 4), hurwitzint(5, 1, 2, 7), hurwitzint(2, 3, 4, 53),
                     hurwitzint(1, 1, 1, 1, half=True))

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

    def test_gcd_of_integers_is_positive(self):
        """The gcd of two integers is their positive integer gcd, whatever their signs"""
        for a, b in ((6, 15), (-6, 15), (6, -15), (-6, -15), (12, 18), (0, -7), (-7, 0), (0, 0)):
            assert hurwitzint(a).gcd_right(b) == gcd(a, b)
            assert hurwitzint(a).gcd_left(b) == gcd(a, b)

    def test_coprime_gcd_is_one(self):
        """Hurwitz integers with coprime norms share no factor, and their gcd comes out as exactly 1"""
        a = hurwitzint(1, 2, 3, 4)  # norm 30
        b = hurwitzint(1, 1, 1, 2)  # norm 7

        assert a.gcd_right(b) == 1
        assert b.gcd_right(a) == 1
        assert a.gcd_left(b) == 1
        assert b.gcd_left(a) == 1

    def test_gcd_does_not_depend_on_argument_order(self):
        """Both orders give the same canonical common factor (this pair once gave 1+j one way and i-j the other)"""
        g = hurwitzint(1, 1, 0, 0)
        b = hurwitzint(1, 2, 3, 4) * g
        c = hurwitzint(5, 1, 2, 7) * g

        assert b.gcd_right(c) == g
        assert c.gcd_right(b) == g

    def test_gcd_right_is_canonical(self):
        """gcd_right does not depend on the argument order, or on units on the left of either argument"""
        g = hurwitzint(3, 5, 7, 9, half=True)
        for x in self.gcd_cofactors:
            for y in self.gcd_cofactors:
                a, b = x * g, y * g
                d = a.gcd_right(b)

                # g is a common right divisor, so it right-divides the greatest one
                assert not divmod(d, g)[1]

                assert b.gcd_right(a) == d
                for u in hurwitzint.UNITS:
                    assert (u * a).gcd_right(b) == d
                    assert a.gcd_right(u * b) == d

    def test_gcd_left_is_canonical(self):
        """gcd_left does not depend on the argument order, or on units on the right of either argument"""
        g = hurwitzint(3, 5, 7, 9, half=True)
        for x in self.gcd_cofactors:
            for y in self.gcd_cofactors:
                a, b = g * x, g * y
                d = a.gcd_left(b)

                # g is a common left divisor, so it left-divides the greatest one
                assert not d.rdivmod(g)[1]

                assert b.gcd_left(a) == d
                for u in hurwitzint.UNITS:
                    assert (a * u).gcd_left(b) == d
                    assert a.gcd_left(b * u) == d

    def test_gcd_is_greatest(self):
        """With coprime N(x) and N(y), x*g and y*g share nothing more than g, so their gcd is exactly g"""
        rng = random.Random(7_000)
        for bound in (3, 10**3, 10**9):
            for _ in range(60):
                g = self.rand_hurwitzint(rng, bound)
                x = self.rand_hurwitzint(rng, bound)
                y = self.rand_hurwitzint(rng, bound)
                while gcd(abs(x), abs(y)) != 1:
                    y = self.rand_hurwitzint(rng, bound)

                # A gcd with 0 is the canonical associate of the other argument, so these name g's canonical associates
                assert (x * g).gcd_right(y * g) == g.gcd_right(0)
                assert (g * x).gcd_left(g * y) == g.gcd_left(0)

    def test_unnormalized_gcd_is_an_associate(self):
        """normalize=False returns whichever gcd the algorithm lands on, which is a unit times the normalized one"""
        g = hurwitzint(3, 5, 7, 9, half=True)
        for x in self.gcd_cofactors:
            for y in self.gcd_cofactors:
                raw = (x * g).gcd_right(y * g, normalize=False)
                assert any(u * raw == (x * g).gcd_right(y * g) for u in hurwitzint.UNITS)

                raw = (g * x).gcd_left(g * y, normalize=False)
                assert any(raw * u == (g * x).gcd_left(g * y) for u in hurwitzint.UNITS)

    def test_module_level_helpers(self):
        """The module-level gcd_left and gcd_right give the same results as the methods"""
        pairs = (
            (hurwitzint(2, 3, 4, 53), self.a_int),
            (hurwitzint(3, 5, 7, 9, half=True) * self.a_int, self.a_int),
            (hurwitzint(12), 18),
        )
        for a, b in pairs:
            assert gcd_left(a, b) == a.gcd_left(b)
            assert gcd_right(a, b) == a.gcd_right(b)

    def test_unsupported_types_raise_type_error(self):
        """A gcd with something that is not a number raises TypeError, on either side"""
        x = self.a_int
        for method in (x.gcd_right, x.gcd_left):
            for other in ("a", None, [1]):
                with pytest.raises(TypeError):
                    method(other)


class TestXgcd(HurwitzIntTests):
    """Tests for xgcd_right and xgcd_left, the gcds with Bezout coefficients"""

    def test_bezout_random(self):
        """
        For random pairs of every size, s*a + t*b == g for xgcd_right, and a*s + b*t == g for xgcd_left,
            where g is the gcd that gcd_right and gcd_left give, normalized or not
        """
        rng = random.Random(12_000)
        for bound_a, bound_b in ((3, 3), (10, 10**4), (10**4, 10**2), (10**12, 10**5), (10**30, 10**12)):
            for _ in range(60):
                a = self.rand_hurwitzint(rng, bound_a)
                b = self.rand_hurwitzint(rng, bound_b)
                for normalize in (True, False):
                    g, s, t = a.xgcd_right(b, normalize=normalize)
                    assert s * a + t * b == g
                    assert g == a.gcd_right(b, normalize=normalize)

                    g, s, t = a.xgcd_left(b, normalize=normalize)
                    assert a * s + b * t == g
                    assert g == a.gcd_left(b, normalize=normalize)

    def test_common_factor(self):
        """
        With coprime N(x) and N(y), x*g and y*g share nothing more than g, so xgcd_right finds g's canonical
            associate as a combination of the two (and xgcd_left does the same for g*x and g*y)
        """
        rng = random.Random(13_000)
        for bound in (3, 10**3, 10**9):
            for _ in range(40):
                g = self.rand_hurwitzint(rng, bound)
                x = self.rand_hurwitzint(rng, bound)
                y = self.rand_hurwitzint(rng, bound)
                while gcd(abs(x), abs(y)) != 1:
                    y = self.rand_hurwitzint(rng, bound)

                d, s, t = (x * g).xgcd_right(y * g)
                assert d == g.gcd_right(0)
                assert s * (x * g) + t * (y * g) == d

                d, s, t = (g * x).xgcd_left(g * y)
                assert d == g.gcd_left(0)
                assert (g * x) * s + (g * y) * t == d

    def test_coprime_gives_inverses(self):
        """
        With coprime norms the gcd is 1, so s*a + t*b == 1 makes s an inverse of a modulo b: s*a - 1 == -t*b is
            a multiple of b on the right (and for xgcd_left, a*s - 1 == -b*t is one on the left)
        """
        rng = random.Random(14_000)
        for bound in (10, 10**6, 10**20):
            for _ in range(40):
                a = self.rand_hurwitzint(rng, bound)
                b = self.rand_hurwitzint(rng, bound)
                while gcd(abs(a), abs(b)) != 1:
                    b = self.rand_hurwitzint(rng, bound)

                g, s, t = a.xgcd_right(b)
                assert g == 1
                assert s * a + t * b == 1
                assert b.divides_right(s * a - 1)

                g, s, t = a.xgcd_left(b)
                assert g == 1
                assert a * s + b * t == 1
                assert b.divides_left(a * s - 1)

    def test_integers(self):
        """For two integers this is the extended Euclidean algorithm: integer coefficients, and the positive gcd"""
        for x, y in ((240, 46), (-240, 46), (46, -240), (17, 5), (12, 18), (0, 9), (-9, 0), (10**20 + 39, 10**18 + 3)):
            for method in (hurwitzint(x).xgcd_right, hurwitzint(x).xgcd_left):
                g, s, t = method(y)

                assert g == gcd(x, y)
                assert s * x + t * y == g
                for coefficient in (s, t):
                    assert coefficient.is_lipschitz
                    assert (coefficient.b, coefficient.c, coefficient.d) == (0, 0, 0)

    def test_units(self):
        """A unit has gcd 1 with anything, on either side"""
        for u in hurwitzint.UNITS:
            for b in (self.a_int, hurwitzint(3, -5, 7, 9, half=True), hurwitzint(0)):
                g, s, t = u.xgcd_right(b)
                assert g == 1
                assert s * u + t * b == 1

                g, s, t = u.xgcd_left(b)
                assert g == 1
                assert u * s + b * t == 1

    def test_zero(self):
        """
        With 0 on one side, the gcd is the canonical associate of the other argument, and its coefficient is
            the unit that makes it so. With 0 on both sides, this is (0, 1, 0)
        """
        for a in (self.a_int, hurwitzint(3, -5, 7, 9, half=True), hurwitzint(-6)):
            g, s, t = a.xgcd_right(0)
            assert (g, t) == (a.gcd_right(0), 0)
            assert s.is_unit
            assert s * a == g

            g, s, t = hurwitzint(0).xgcd_right(a)
            assert (g, s) == (a.gcd_right(0), 0)
            assert t.is_unit
            assert t * a == g

            g, s, t = a.xgcd_left(0)
            assert (g, t) == (a.gcd_left(0), 0)
            assert s.is_unit
            assert a * s == g

            g, s, t = hurwitzint(0).xgcd_left(a)
            assert (g, s) == (a.gcd_left(0), 0)
            assert t.is_unit
            assert a * t == g

        for normalize in (True, False):
            assert hurwitzint(0).xgcd_right(0, normalize=normalize) == (0, 1, 0)
            assert hurwitzint(0).xgcd_left(0, normalize=normalize) == (0, 1, 0)

    def test_int_and_float_arguments(self):
        """An int or float argument works like the hurwitzint it equals"""
        a = hurwitzint(3, -5, 7, 9, half=True) * hurwitzint(1, 1, 0, 0)
        for m in (2, -2, 82, 41.0, 6.9):
            assert a.xgcd_right(m) == a.xgcd_right(hurwitzint(m))
            assert a.xgcd_left(m) == a.xgcd_left(hurwitzint(m))

    def test_module_level_helpers(self):
        """The module-level xgcd_left and xgcd_right give the same results as the methods, and quatint exports them"""
        pairs = (
            (hurwitzint(2, 3, 4, 53), self.a_int),
            (hurwitzint(3, 5, 7, 9, half=True) * self.a_int, self.a_int),
            (hurwitzint(12), 18),
        )
        for a, b in pairs:
            assert xgcd_left(a, b) == a.xgcd_left(b)
            assert xgcd_right(a, b) == a.xgcd_right(b)

        assert quatint.xgcd_left is xgcd_left
        assert quatint.xgcd_right is xgcd_right

    def test_unsupported_types_raise_type_error(self):
        """An extended gcd with something that is not a number raises TypeError, on either side"""
        x = self.a_int
        for method in (x.xgcd_right, x.xgcd_left):
            for other in ("a", None, [1]):
                with pytest.raises(TypeError):
                    method(other)


class TestModuleLevelHelpers(HurwitzIntTests):
    """Tests for the module-level rdivmod, gcd_left, gcd_right, xgcd_left and xgcd_right with a plain number first"""

    helpers = (rdivmod, gcd_left, gcd_right, xgcd_left, xgcd_right)

    def test_examples(self):
        """Two plain integers give their usual gcd, and a number divides like the hurwitzint it equals"""
        assert gcd_right(12, 18) == gcd_left(-12, 18.0) == 6

        for g, s, t in (xgcd_right(240, 46), xgcd_left(240, 46)):
            assert g == 2
            assert s * 240 + t * 46 == 2

        # 30 is the norm of 1+2i+3j+4k, so it is (1+2i+3j+4k) times its conjugate
        assert rdivmod(30, hurwitzint(1, 2, 3, 4)) == (hurwitzint(1, -2, -3, -4), 0)

    def test_number_first(self):
        """An int or float first works like the hurwitzint it equals, whatever comes second"""
        for helper in self.helpers:
            for a in (12, -7, 0, 6.9, -2.0):
                for b in (hurwitzint(1, 1, 0, 0), hurwitzint(3, -5, 7, 9, half=True), 18, 4.5):
                    assert helper(a, b) == helper(hurwitzint(a), b)

    def test_unsupported_first_argument(self):
        """Anything else first raises TypeError, in both builds"""
        for helper in self.helpers:
            for a in ("a", None, [1], complex(1, 2)):
                with pytest.raises(TypeError):
                    helper(a, hurwitzint(1, 2, 3, 4))


class TestResidue(HurwitzIntTests):
    """Tests for _residue_numerators, x % m worked out directly for an integer m, which inv_mod and pow reduce with"""

    @staticmethod
    def residue(x: hurwitzint, m: int) -> hurwitzint:
        """The remainder of x modulo m > 0, from _residue_numerators"""
        return hurwitzint(*quatint.quat._residue_numerators(*x, m), half=True)

    def test_matches_brute_force(self):
        """
        The residue is the brute-forced one, the least norm with ties going to the largest tuple, for every small
            value modulo the small moduli that make the most ties, and random values of every size modulo bigger ones
        """
        for m in (2, 3, 4):
            for x in self.wide_search_values(2):
                assert self.residue(x, m) == self.canonical_remainder(x, hurwitzint(m))

        rng = random.Random(17_000)
        for bound in (10, 10**4, 10**30):
            for _ in range(30):
                x = self.rand_hurwitzint(rng, bound)
                for m in (1, 5, 6, 7, 12, 97, 10**9 + 7, 2**64):
                    assert self.residue(x, m) == self.canonical_remainder(x, hurwitzint(m))

    def test_matches_division(self):
        """
        It is the very remainder that division by m leaves, on either side, and dividing by -m too, so it is what
            x % m gives, and pow(x, 1, m) == x % m. For every small value modulo small m (where ties are common), and
            random values of every size modulo bigger ones
        """
        for m in (1, 2, 3, 4, 6, 7):
            for x in self.wide_search_values(2):
                r = self.residue(x, m)
                assert x % m == x.rmod(m) == x % -m == pow(x, 1, m) == r

        rng = random.Random(17_002)
        for bound in (10, 10**4, 10**30):
            for _ in range(30):
                x = self.rand_hurwitzint(rng, bound)
                for m in (5, 12, 97, 10**9 + 7, 2**64):
                    assert x % m == x.rmod(m) == self.residue(x, m)

    def test_congruent_values(self):
        """Values congruent modulo m have the very same residue"""
        rng = random.Random(17_001)
        shifts = (hurwitzint(1), hurwitzint(-1), hurwitzint(0, 1, 0, 0), hurwitzint(1, 1, 1, 1, half=True),
                  hurwitzint(-1, 1, -1, 1, half=True))
        for _ in range(100):
            x = self.rand_hurwitzint(rng, 10)
            m = rng.choice((2, 3, 4, 6))
            r = self.residue(x, m)
            assert hurwitzint(m).divides_right(x - r)
            for h in shifts:
                assert self.residue(x + m * h, m) == r


class TestInvMod(HurwitzIntTests):
    """Tests for inv_mod, the inverse modulo an integer, on both sides"""

    def test_examples(self):
        """Some inverses worked out by hand"""
        # (1+i) * (-1+i) == -2, which is 1 mod 3
        assert hurwitzint(1, 1, 0, 0).inv_mod(3) == hurwitzint(-1, 1, 0, 0)

        # 3 * -2 == -6, which is 1 mod 7. pow(3, -1, 7) is 5, but -2 is the residue of least norm
        assert hurwitzint(3).inv_mod(7) == -2

        # A unit's inverse is its conjugate, which is the least norm there is, but mod 2 so is its negative. Then the
        #   larger numerator tuple wins: (1+i+j+k)/2 over (-1-i-j-k)/2
        u = hurwitzint(-1, 1, 1, 1, half=True)
        assert u.inv_mod(5) == ~u == hurwitzint(-1, -1, -1, -1, half=True)
        assert u.inv_mod(2) == -~u == hurwitzint(1, 1, 1, 1, half=True)

        # Everything is 0 mod 1, so 0 is everything's inverse
        assert hurwitzint(5, 3, 1, 2).inv_mod(1) == hurwitzint(0).inv_mod(-1) == 0

    def test_inverse_on_both_sides(self):
        """For random x and m, x times its inverse is 1 modulo m on either side, whenever N(x) and m are coprime"""
        rng = random.Random(16_000)
        invertible = 0
        for bound in (3, 10**4, 10**12):
            for _ in range(80):
                x = self.rand_hurwitzint(rng, bound)
                m = rng.choice((2, 3, 4, 5, 7, 12, 97, 1000, 10**9 + 7, 2**61 - 1))
                if gcd(abs(x), m) != 1:
                    with pytest.raises(ValueError, match="not invertible mod"):
                        x.inv_mod(m)

                    continue

                invertible += 1
                y = x.inv_mod(m)
                assert hurwitzint(m).divides_right(x * y - 1)
                assert hurwitzint(m).divides_right(y * x - 1)
                assert y == y % m  # Reduced already

        assert invertible > 100

    def test_canonical(self):
        """Congruent values have the very same inverse, the canonical residue of the inverse's class"""
        rng = random.Random(16_001)
        for _ in range(60):
            x = self.rand_hurwitzint(rng, 100)
            m = rng.choice((2, 3, 4, 6, 7, 10))
            if gcd(abs(x), m) != 1:
                continue

            # conj(x) / N(x) is the inverse, so conj(x) times the inverse of N(x) mod m is an inverse mod m
            y = x.inv_mod(m)
            assert y == (x.conjugate() * pow(abs(x), -1, m)) % m
            for h in (hurwitzint(1), hurwitzint(1, 1, 1, 1, half=True), self.rand_hurwitzint(rng, 10)):
                assert (x + m * h).inv_mod(m) == y

    def test_not_invertible(self):
        """There is no inverse when N(x) shares a factor with the modulus, as for 0, or 1+i mod 2"""
        for x, m in ((hurwitzint(0), 5), (hurwitzint(1, 1, 0, 0), 2), (hurwitzint(3), 6), (hurwitzint(1, 2, 3, 4), 10)):
            with pytest.raises(ValueError, match="not invertible mod"):
                x.inv_mod(m)

    def test_moduli(self):
        """A float modulus is truncated, and a negative one, or a real hurwitzint, works like the positive int"""
        x = hurwitzint(3, -5, 7, 9, half=True)  # norm 41
        expected = x.inv_mod(7)
        for mod in (7.0, 7.9, -7, -7.5, hurwitzint(7), hurwitzint(-7)):
            assert x.inv_mod(mod) == expected

    def test_bad_moduli(self):
        """A modulus of 0 raises ZeroDivisionError, a hurwitzint off the real axis ValueError, and the rest TypeError"""
        x = hurwitzint(3, -5, 7, 9, half=True)
        for zero in (0, 0.0, 0.9, hurwitzint(0)):
            with pytest.raises(ZeroDivisionError):
                x.inv_mod(zero)

        for mod in (hurwitzint(1, 1, 0, 0), hurwitzint(3, 1, 1, 1, half=True)):
            with pytest.raises(ValueError, match="integer modulus"):
                x.inv_mod(mod)

        for mod in ("7", None, [7], complex(7, 0)):
            with pytest.raises(TypeError):
                x.inv_mod(mod)


class TestPowMod(HurwitzIntTests):
    """Tests for pow(x, e, mod), powers modulo an integer"""

    def test_examples(self):
        """Some powers worked out by hand"""
        # i*i == -1, which has the least norm there is, so it is its own residue
        assert pow(hurwitzint(0, 1, 0, 0), 2, 5) == -1

        # 81 == 11*7 + 4, but -3 is the residue of least norm (Python's pow gives 4)
        assert pow(hurwitzint(3), 4, 7) == -3

        # (1+i)**2 == 2i, and 2i == -i (mod 3)
        assert pow(hurwitzint(1, 1, 0, 0), 2, 3) == hurwitzint(0, -1, 0, 0)

    def test_matches_reducing_the_power(self):
        """pow(x, e, m) == (x**e) % m, as for an int, for random x, e and m"""
        rng = random.Random(18_000)
        for bound in (3, 10**4, 10**12):
            for _ in range(40):
                x = self.rand_hurwitzint(rng, bound)
                e = rng.randint(0, 12)
                m = rng.choice((1, 2, 3, 4, 6, 7, 10, 97, 10**9 + 7))
                assert pow(x, e, m) == (x**e) % m

    def test_congruent_bases(self):
        """Congruent bases have the very same powers"""
        rng = random.Random(18_001)
        for _ in range(60):
            x = self.rand_hurwitzint(rng, 10)
            m = rng.choice((2, 3, 4, 6))
            h = self.rand_hurwitzint(rng, 10)
            for e in (0, 1, 2, 5):
                assert pow(x + m * h, e, m) == pow(x, e, m)

    def test_power_laws(self):
        """Modular powers combine like powers do, for random values and exponents"""
        rng = random.Random(18_002)
        for _ in range(60):
            x = self.rand_hurwitzint(rng, 10**6)
            k = rng.choice((2, 3, 5, 12, 97, 10**9 + 7))
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            assert pow(x, a + b, k) == pow(pow(x, a, k) * pow(x, b, k), 1, k)
            assert pow(x, a * b, k) == pow(pow(x, a, k), b, k)

    def test_huge_exponents(self):
        """Every bit of a huge exponent counts, even past what a float holds (doubles round past 2**53)"""
        x = hurwitzint(1, 2, 3, 4)  # Of norm 30, so invertible mod 97, so x**(e+1) == x**e would mean x == 1
        for e in (2**64, 2**1100):
            assert pow(x, e + 1, 97) == pow(pow(x, e, 97) * x, 1, 97)
            assert pow(x, e + 1, 97) != pow(x, e, 97)

    def test_exponent_zero(self):
        """x**0 is 1, reduced: 1 itself for any modulus but 1, where everything is 0"""
        for x in (hurwitzint(0), hurwitzint(1, 2, 3, 4), hurwitzint(3, -5, 7, 9, half=True)):
            assert pow(x, 0, 1) == 0
            for m in (2, 3, 97, -5):
                assert pow(x, 0, m) == 1

    def test_negative_exponents(self):
        """A negative exponent takes powers of the inverse modulo m, so it needs one to exist"""
        rng = random.Random(18_003)
        for _ in range(60):
            x = self.rand_hurwitzint(rng, 10**4)
            m = rng.choice((3, 5, 7, 12, 97))
            if gcd(abs(x), m) != 1:
                with pytest.raises(ValueError, match="not invertible mod"):
                    pow(x, -1, m)

                continue

            for e in (1, 2, 7):
                assert pow(x, -e, m) == pow(x.inv_mod(m), e, m)
                assert pow(pow(x, -e, m) * pow(x, e, m), 1, m) == 1

    def test_moduli(self):
        """A float modulus is truncated, and a negative one, or a real hurwitzint, works like the positive int"""
        x = hurwitzint(3, -5, 7, 9, half=True)
        expected = pow(x, 5, 7)
        for mod in (7.0, 7.9, -7, hurwitzint(7), hurwitzint(-7)):
            assert pow(x, 5, mod) == expected

        assert pow(x, 5, None) == x**5  # As for an int, None is no modulus

    def test_bad_moduli(self):
        """A modulus of 0 raises ZeroDivisionError, a hurwitzint off the real axis ValueError, and the rest TypeError"""
        x = hurwitzint(3, -5, 7, 9, half=True)
        for zero in (0, 0.0, hurwitzint(0)):
            with pytest.raises(ZeroDivisionError):
                pow(x, 2, zero)

        with pytest.raises(ValueError, match="integer modulus"):
            pow(x, 2, hurwitzint(1, 1, 0, 0))

        for mod in ("7", [7], complex(7, 0)):
            with pytest.raises(TypeError):
                pow(x, 2, mod)


class TestContent(HurwitzIntTests):
    """Tests for content, the largest integer that divides a Hurwitz integer"""

    def test_examples(self):
        """Some values worked out by hand, including multiples of 1+i+j+k, which is 2 times the unit (1+i+j+k)/2"""
        assert hurwitzint(0).content() == 0
        assert hurwitzint(6).content() == 6
        assert hurwitzint(-6).content() == 6
        assert hurwitzint(1, 2, 3, 4).content() == 1
        assert hurwitzint(2, 4, 6, 8).content() == 2
        assert hurwitzint(6, 2, 4, 0).content() == 2
        assert hurwitzint(1, 1, 1, 1).content() == 2
        assert hurwitzint(3, 3, 3, 3).content() == 6
        assert hurwitzint(3, 5, 7, 9, half=True).content() == 1
        assert hurwitzint(3, 3, 3, 3, half=True).content() == 3

    def test_content_is_largest(self):
        """Every x is its content times a Hurwitz integer of content 1, and m*x has |m| times the content of x"""
        rng = random.Random(8_000)
        for bound in (3, 10**4, 10**30):
            for _ in range(100):
                x = self.rand_hurwitzint(rng, bound)
                m = x.content()
                primitive = x // m

                assert m * primitive == x
                assert primitive.content() == 1

                k = rng.choice((2, -3, 12, 10**9))
                assert (k * x).content() == abs(k) * m


class TestProd(HurwitzIntTests):
    """Tests for prod_right and prod_left, which multiply in opposite orders"""

    def test_order(self):
        """prod_right multiplies left to right like math.prod, and prod_left puts each new factor on the left"""
        a, b, c = self.a_int, self.b_int, hurwitzint(3, 5, 7, 9, half=True)

        assert prod_right((a, b, c)) == a * b * c
        assert prod_left((a, b, c)) == c * b * a
        assert prod_right((a, b, c)) != prod_left((a, b, c))

    def test_start(self):
        """The start value ends up on the left for prod_right, and on the right for prod_left"""
        a, b, c = self.a_int, self.b_int, hurwitzint(3, 5, 7, 9, half=True)

        assert prod_right((a, b), start=c) == c * a * b
        assert prod_left((a, b), start=c) == b * a * c

    def test_empty(self):
        """With nothing to multiply, the product is the start value, which defaults to 1"""
        assert prod_right(()) == 1
        assert prod_left(()) == 1
        assert prod_right((), start=self.a_int) == self.a_int
        assert prod_left((), start=self.a_int) == self.a_int


class TestPrimeOfNorm(HurwitzIntTests):
    """Tests for hurwitzint.prime_of_norm, a fixed Hurwitz prime for each rational prime norm"""

    # Every prime below 2000, and a big one of each kind the square root search handles: p % 4 == 3, then p % 4 == 1
    primes = (*(p for p in range(2000) if isprime(p)), 2**127 - 1, 10**40 + 121)

    def test_norm_and_factors_of_p(self):
        """The prime has norm p, so it and its conjugate multiply to p, in either order"""
        assert isprime(10**40 + 121)
        for p in self.primes:
            for direction in ("right", "left"):
                prime = hurwitzint.prime_of_norm(p, direction=direction)

                assert abs(prime) == p
                assert prime.conjugate() * prime == p
                assert prime * prime.conjugate() == p

    def test_canonical(self):
        """Like the factorizations' primes, the right one is the largest of all u*P, and the left one of all P*u"""
        for p in self.primes[:100]:
            right = hurwitzint.prime_of_norm(p)
            left = hurwitzint.prime_of_norm(p, direction="left")

            assert all(tuple(u * right) <= tuple(right) for u in hurwitzint.UNITS)
            assert all(tuple(left * u) <= tuple(left) for u in hurwitzint.UNITS)

    def test_fixed_choices(self):
        """The choice depends on nothing but p, so pin some of them"""
        assert hurwitzint.prime_of_norm(2) == hurwitzint(1, 1, 0, 0)
        assert hurwitzint.prime_of_norm(2, direction="left") == hurwitzint(1, 1, 0, 0)
        assert hurwitzint.prime_of_norm(3) == hurwitzint(3, 1, -1, -1, half=True)
        assert hurwitzint.prime_of_norm(3, direction="left") == hurwitzint(3, 1, -1, 1, half=True)
        assert hurwitzint.prime_of_norm(5) == hurwitzint(2, 0, -1, 0)
        assert hurwitzint.prime_of_norm(13) == hurwitzint(3, 0, 2, 0)

    def test_uv_is_least(self):
        """_uv_for_prime finds the least u, and then the least v, with 1 + u^2 + v^2 divisible by p (by brute force)"""
        for p in (q for q in range(200) if isprime(q)):
            least = next((u, v) for u in range(p) for v in range(p) if (1 + u * u + v * v) % p == 0)
            assert quatint.quat._uv_for_prime(p) == least

    def test_not_prime(self):
        """Anything but a prime raises ValueError (a composite could leave the square root search looping forever)"""
        for n in (-7, -2, 0, 1, 4, 9, 15, 21, 561, 2**127 + 1):
            with pytest.raises(ValueError, match="not prime"):
                hurwitzint.prime_of_norm(n)

    def test_float_is_truncated(self):
        """Like everywhere else a float meets a hurwitzint, it is truncated with int()"""
        assert hurwitzint.prime_of_norm(5.0) == hurwitzint.prime_of_norm(5)
        assert hurwitzint.prime_of_norm(5.9) == hurwitzint.prime_of_norm(5)
        assert hurwitzint.prime_of_norm(3.5, direction="left") == hurwitzint.prime_of_norm(3, direction="left")

    def test_unsupported_types_and_directions(self):
        """A non-number raises TypeError in both builds, and an unknown direction raises ValueError"""
        for bad in ("5", None, Fraction(5), hurwitzint(5)):
            with pytest.raises(TypeError):
                hurwitzint.prime_of_norm(bad)

        with pytest.raises(ValueError, match="direction"):
            hurwitzint.prime_of_norm(5, direction="up")


class TestFactorRightDetail(HurwitzIntTests):
    """Tests for factor_right_detail"""

    def test_main(self):
        """Validate factor works as expected."""
        self.assert_factoring(self.b_int, self.b_int.factor_right_detail())

    def test_examples(self):
        """Validate factor works as expected for some given examples."""
        n = hurwitzint(2, 3, 4, 53)
        self.assert_factoring(n, n.factor_right_detail())

        # This comes out unsorted by norm if primes are extracted smallest norm first
        n = hurwitzint(1, 1, 1, 6)
        self.assert_factoring(n, n.factor_right_detail())

        # This once broke a metacommutation swap, back when primes were sorted by swapping them
        n = hurwitzint(1, 1, 2, 15)
        self.assert_factoring(n, n.factor_right_detail())

        n = hurwitzint(17 * 31, 0, 0, 0)
        self.assert_factoring(n, n.factor_right_detail())

    def test_wide_search(self):
        """Validate factoring every small Lipschitz and half-integer Hurwitz integer, negative components included"""
        for n in self.wide_search_values():
            self.assert_factoring(n, n.factor_right_detail())

    def test_normal_form(self):
        """A unit on the left only changes the leading unit and never the primes, so this is a normal form"""
        for n in (self.b_int, hurwitzint(2, 3, 4, 53), hurwitzint(1, 1, 2, 15), hurwitzint(6, 2, 4, 0),
                  hurwitzint(4, 5, 22, 50), hurwitzint(3, 5, 7, 9, half=True) * self.a_int):
            factors = n.factor_right_detail()
            for u in hurwitzint.UNITS:
                moved = (u * n).factor_right_detail()

                assert moved.content == factors.content
                assert moved.primes == factors.primes
                assert moved.unit == u * factors.unit

    def test_random_products(self):
        """Factor seeded random products of bigger Hurwitz integers (norms up to about 10**20), some with content"""
        rng = random.Random(9_000)
        for bound in (10, 10**3, 10**5):
            for _ in range(15):
                n = self.rand_hurwitzint(rng, bound) * self.rand_hurwitzint(rng, bound)
                if rng.random() < 0.3:
                    n *= rng.randint(2, 30)

                self.assert_factoring(n, n.factor_right_detail())

    def assert_factoring(self, n: hurwitzint, factors: NonCommutativeFactorization):
        """Validate everything about the factoring is correct"""
        ans = factors.prod_right()

        self.assert_equal(n, ans)
        assert factors.prod() == n  # prod() multiplies a right factorization with prod_right

        # Validate the primes are sorted by norm
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

        factors = n.factor_right()

        assert n.factor_right_detail().content > 1
        assert len(n.factor_right_detail().primes) > 1

        ans = prod_right(factors)

        self.assert_equal(n, ans)

    def test_zero_units_and_primes(self):
        """Zero, a unit, or a prime (anything with a prime norm) factors as just itself"""
        primes = (hurwitzint(1, 1, 0, 0), hurwitzint(3, 1, 1, 1, half=True), hurwitzint(1, 1, 1, 2),
                  hurwitzint(6, 0, 0, 1))
        for n in (hurwitzint(0), *hurwitzint.UNITS, *primes):
            assert n.factor_right() == (n,)

    def test_content_is_factored(self):
        """The content is factored too, so every factor is a Hurwitz prime, even for a plain integer"""
        for n, norms in ((hurwitzint(527), [17, 17, 31, 31]), (hurwitzint(2), [2, 2]),
                         (hurwitzint(-12), [2, 2, 2, 2, 3, 3]), (hurwitzint(6, 2, 4, 0), [2, 2, 2, 7])):
            factors = n.factor_right()

            assert prod_right(factors) == n
            assert [abs(p) for p in factors] == norms

    def test_random_products(self):
        """For seeded random products, the factors multiply back with prod_right, and every one is a prime"""
        rng = random.Random(9_002)
        for bound in (10, 10**3):
            for _ in range(20):
                n = self.rand_hurwitzint(rng, bound) * self.rand_hurwitzint(rng, bound) * rng.randint(1, 12)
                factors = n.factor_right()

                assert prod_right(factors) == n
                assert all(isprime(abs(p)) for p in factors)


class TestFactorLeftDetail(HurwitzIntTests):
    """Tests for factor_left_detail"""

    def test_main(self):
        """Validate factor works as expected."""
        self.assert_factoring(self.b_int, self.b_int.factor_left_detail())

    def test_examples(self):
        """Validate factor works as expected for some given examples."""
        n = hurwitzint(2, 3, 4, 53)
        self.assert_factoring(n, n.factor_left_detail())

        # This comes out unsorted by norm if primes are extracted smallest norm first
        n = hurwitzint(1, 1, 1, 6)
        self.assert_factoring(n, n.factor_left_detail())

        # This once broke a metacommutation swap, back when primes were sorted by swapping them
        n = hurwitzint(1, 1, 2, 15)
        self.assert_factoring(n, n.factor_left_detail())

        n = hurwitzint(17 * 31, 0, 0, 0)
        self.assert_factoring(n, n.factor_left_detail())

    def test_wide_search(self):
        """Validate factoring every small Lipschitz and half-integer Hurwitz integer, negative components included"""
        for n in self.wide_search_values():
            self.assert_factoring(n, n.factor_left_detail())

    def test_normal_form(self):
        """A unit on the right only changes the trailing unit and never the primes, so this is a normal form"""
        for n in (self.b_int, hurwitzint(2, 3, 4, 53), hurwitzint(1, 1, 2, 15), hurwitzint(6, 2, 4, 0),
                  hurwitzint(4, 5, 22, 50), hurwitzint(3, 5, 7, 9, half=True) * self.a_int):
            factors = n.factor_left_detail()
            for u in hurwitzint.UNITS:
                moved = (n * u).factor_left_detail()

                assert moved.content == factors.content
                assert moved.primes == factors.primes
                assert moved.unit == factors.unit * u

    def test_random_products(self):
        """Factor seeded random products of bigger Hurwitz integers (norms up to about 10**20), some with content"""
        rng = random.Random(9_001)
        for bound in (10, 10**3, 10**5):
            for _ in range(15):
                n = self.rand_hurwitzint(rng, bound) * self.rand_hurwitzint(rng, bound)
                if rng.random() < 0.3:
                    n *= rng.randint(2, 30)

                self.assert_factoring(n, n.factor_left_detail())

    def assert_factoring(self, n: hurwitzint, factors: NonCommutativeFactorization):
        """Validate everything about the factoring is correct"""
        ans = factors.prod_left()

        self.assert_equal(n, ans)
        assert factors.prod() == n  # prod() multiplies a left factorization with prod_left

        # Validate the primes are sorted by norm
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

        factors = n.factor_left()

        assert n.factor_left_detail().content > 1
        assert len(n.factor_left_detail().primes) > 1

        ans = prod_left(factors)

        self.assert_equal(n, ans)

    def test_zero_units_and_primes(self):
        """Zero, a unit, or a prime (anything with a prime norm) factors as just itself"""
        primes = (hurwitzint(1, 1, 0, 0), hurwitzint(3, 1, 1, 1, half=True), hurwitzint(1, 1, 1, 2),
                  hurwitzint(6, 0, 0, 1))
        for n in (hurwitzint(0), *hurwitzint.UNITS, *primes):
            assert n.factor_left() == (n,)

    def test_content_is_factored(self):
        """The content is factored too, so every factor is a Hurwitz prime, even for a plain integer"""
        for n, norms in ((hurwitzint(527), [17, 17, 31, 31]), (hurwitzint(2), [2, 2]),
                         (hurwitzint(-12), [2, 2, 2, 2, 3, 3]), (hurwitzint(6, 2, 4, 0), [2, 2, 2, 7])):
            factors = n.factor_left()

            assert prod_left(factors) == n
            assert [abs(p) for p in factors] == norms

    def test_random_products(self):
        """For seeded random products, the factors multiply back with prod_left, and every one is a prime"""
        rng = random.Random(9_003)
        for bound in (10, 10**3):
            for _ in range(20):
                n = self.rand_hurwitzint(rng, bound) * self.rand_hurwitzint(rng, bound) * rng.randint(1, 12)
                factors = n.factor_left()

                assert prod_left(factors) == n
                assert all(isprime(abs(p)) for p in factors)


class TestExpandContent(HurwitzIntTests):
    """Tests for NonCommutativeFactorization.expand_content, which factors the content into Hurwitz primes too"""

    @staticmethod
    def detail(n: hurwitzint, direction: str) -> NonCommutativeFactorization:
        """The detail factorization of n in the given direction"""
        return n.factor_right_detail() if direction == "right" else n.factor_left_detail()

    @staticmethod
    def assert_expanded(n: hurwitzint, factors: NonCommutativeFactorization):
        """Validate an expanded factorization: exact, content 1, and its primes sorted by norm and each canonical"""
        assert factors.prod() == n
        assert factors.content == 1
        assert abs(factors.unit) == 1

        norms = [abs(p) for p in factors.primes]
        assert norms == sorted(norms)
        assert all(isprime(norm) for norm in norms)

        # Canonical like the other primes: the largest u*P in a right factorization, and the largest P*u in a left one
        for p in factors.primes:
            if factors.direction == "right":
                assert all(tuple(u * p) <= tuple(p) for u in hurwitzint.UNITS)
            else:
                assert all(tuple(p * u) <= tuple(p) for u in hurwitzint.UNITS)

    def test_examples(self):
        """The choice of primes is fixed, so pin some of them"""
        one_plus_i = hurwitzint(1, 1, 0, 0)
        for direction in ("right", "left"):
            factors = self.detail(hurwitzint(2), direction).expand_content()
            assert factors == NonCommutativeFactorization(content=1, unit=hurwitzint(0, -1, 0, 0),
                                                          primes=(one_plus_i, one_plus_i), direction=direction)

        # 527 = 17 * 31, and each of those is a prime of norm p times its conjugate
        factors = hurwitzint(527).factor_right_detail().expand_content()
        assert factors.unit == 1
        assert factors.primes == (hurwitzint(4, 0, 1, 0), hurwitzint(4, 0, -1, 0),
                                  hurwitzint(5, -2, 1, 1), hurwitzint(5, 2, -1, -1))

    def test_integers(self):
        """An integer's primes all come from its content, a pair of norm p for each prime factor p"""
        for m in range(-60, 61):
            if not m:
                continue

            for direction in ("right", "left"):
                factors = self.detail(hurwitzint(m), direction).expand_content()

                self.assert_expanded(hurwitzint(m), factors)
                assert [abs(p) for p in factors.primes] == [
                    q for q, e in sorted(factorint(abs(m)).items()) for _ in range(2 * e)
                ]

    def test_content_meets_primes_of_its_norm(self):
        """Multiply every Hurwitz integer of norm p (2, 3 or 5) by p, so the content's primes meet ones of equal norm"""
        for n in self.wide_search_values():
            p = abs(n)
            if p in {2, 3, 5}:
                for direction in ("right", "left"):
                    self.assert_expanded(p * n, self.detail(p * n, direction).expand_content())

    def test_random_products(self):
        """Seeded random products with content up to 60, so with repeated primes, in both directions"""
        rng = random.Random(9_004)
        for bound in (3, 10, 10**3):
            for _ in range(25):
                n = self.rand_hurwitzint(rng, bound) * self.rand_hurwitzint(rng, bound) * rng.randint(2, 60)
                for direction in ("right", "left"):
                    factors = self.detail(n, direction).expand_content()

                    self.assert_expanded(n, factors)
                    assert factors.expand_content() is factors  # Nothing is left to expand

    def test_normal_form(self):
        """A unit on the same side as the unit only changes the unit, and never the primes"""
        for n in (hurwitzint(6), hurwitzint(6, 2, 4, 0), 12 * hurwitzint(1, 1, 1, 0),
                  30 * hurwitzint(3, 5, 7, 9, half=True)):
            right = n.factor_right_detail().expand_content()
            left = n.factor_left_detail().expand_content()
            for u in hurwitzint.UNITS:
                moved = (u * n).factor_right_detail().expand_content()
                assert moved.primes == right.primes
                assert moved.unit == u * right.unit

                moved = (n * u).factor_left_detail().expand_content()
                assert moved.primes == left.primes
                assert moved.unit == left.unit * u

    def test_nothing_to_expand(self):
        """Zero, and anything with content 1, come back as the very same factorization"""
        for n in (hurwitzint(0), hurwitzint(1, 1, 0, 0), self.b_int, *hurwitzint.UNITS):
            for direction in ("right", "left"):
                factors = self.detail(n, direction)
                assert factors.expand_content() is factors

    def test_known_factors(self):
        """Given the content's factors, the result is the same, and a content too big to factor quickly is fine"""
        rng = random.Random(9_005)
        for _ in range(20):
            n = self.rand_hurwitzint(rng, 10) * rng.randint(2, 500)
            for direction in ("right", "left"):
                factors = self.detail(n, direction)
                assert factors.expand_content(factors=factorint(factors.content)) == factors.expand_content()

        # Two 18-digit primes. Factoring their product takes sympy seconds, but with the factors given this is quick.
        p, q = 100000000000000003, 300000000000000011
        assert isprime(p)
        assert isprime(q)

        n = p * q * hurwitzint(3, 5, 7, 9, half=True)
        for direction in ("right", "left"):
            factors = self.detail(n, direction)
            assert factors.content == p * q
            self.assert_expanded(n, factors.expand_content(factors={p: 1, q: 1}))

    def test_bad_factors(self):
        """The factors given have to be the content's prime factorization, with nothing but ints, in both builds"""
        factors = hurwitzint(527).factor_right_detail()  # 17 * 31
        for bad in ({17: 1}, {17: 1, 31: 2}, {527: 1}, {-17: 1, -31: 1}):
            with pytest.raises(ValueError, match=r"content|prime"):
                factors.expand_content(factors=bad)

        # A negative exponent, even where the float product rounds to exactly the content, as 2**60 * 3**-1 does here
        m = 384307168202282304
        assert m == 2**60 * 3**-1
        for n, bad in ((hurwitzint(527), {17: 2, 31: -1}), (hurwitzint(m), {2: 60, 3: -1})):
            for direction in ("right", "left"):
                with pytest.raises(ValueError, match="negative"):
                    self.detail(n, direction).expand_content(factors=bad)

        for bad in ({17.0: 1, 31: 1}, {17: 1.0, 31: 1}):
            with pytest.raises(TypeError):
                factors.expand_content(factors=bad)

        # It has to be a dict too, in both builds, though a dict subclass like Counter is fine
        for bad in (MappingProxyType({17: 1, 31: 1}), [(17, 1), (31, 1)], "17", 527):
            with pytest.raises(TypeError):
                factors.expand_content(factors=bad)

        assert factors.expand_content(factors=Counter({17: 1, 31: 1})) == factors.expand_content()

        # A zero exponent changes nothing
        assert factors.expand_content(factors={2: 0, 17: 1, 31: 1}) == factors.expand_content()

        # 0 has no factorization, and the only one of 1 is empty
        with pytest.raises(ValueError, match="content"):
            hurwitzint(0).factor_right_detail().expand_content(factors={})

        with pytest.raises(ValueError, match="content"):
            hurwitzint(1).factor_right_detail().expand_content(factors={5: 1})

        factors = hurwitzint(1).factor_right_detail()
        assert factors.expand_content(factors={}) is factors


class TestFactorintTypes(HurwitzIntTests):
    """sympy.factorint can answer with integers that are not ints, which the mypyc build rejects where an int goes"""

    def test_non_int_factors(self, monkeypatch: pytest.MonkeyPatch):
        """
        Factoring gives the same answers when factorint's primes and exponents are not ints.

        With gmpy2 or python-flint installed, sympy 1.13 and later return their mpz or fmpz for some factors
            (the first factorint(7 * 104729**3) comes back as {7: 1, mpz(104729): mpz(3)}).
        """

        class NotInt:
            # Like mpz: equal to, ordered and hashed like its value, and converted by int(), but not an int
            def __init__(self, value: int):
                self.value = value

            def __int__(self) -> int:
                return self.value

            def __index__(self) -> int:
                return self.value

            def __eq__(self, other: object) -> bool:
                return self.value == int(other)

            def __hash__(self) -> int:
                return hash(self.value)

            def __lt__(self, other: object) -> bool:
                return self.value < int(other)

            def __gt__(self, other: object) -> bool:
                return self.value > int(other)

        # Primes of the norm, of the content, and of both
        values = (hurwitzint(2, 3, 4, 53), hurwitzint(527), 30 * hurwitzint(3, 5, 7, 9, half=True))
        expected = [(n.factor_right_detail(), n.factor_left_detail(), n.factor_right(), n.factor_left())
                    for n in values]

        real = quatint.quat.factorint
        monkeypatch.setattr(quatint.quat, "factorint", lambda n: {NotInt(p): NotInt(e) for p, e in real(n).items()})
        assert all(type(p) is NotInt for p in quatint.quat.factorint(30))

        for n, answers in zip(values, expected, strict=True):
            assert (n.factor_right_detail(), n.factor_left_detail(), n.factor_right(), n.factor_left()) == answers


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

    def test_repr_of_k(self):
        """Like Python's 2j, a whole multiple of k shows without parentheses, and k or -k without the 1"""
        assert repr(hurwitzint(0, 0, 0, 1)) == "k"
        assert repr(hurwitzint(0, 0, 0, -1)) == "-k"
        assert repr(hurwitzint(0, 0, 0, 12)) == "12k"

    def test_repr_signs(self):
        """A negative part shows a minus sign in place of the plus, in Lipschitz and half-integer values alike"""
        assert repr(hurwitzint(-1, -3, 5, -7, half=True)) == "(-1-3i+5j-7k)/2"
        assert repr(hurwitzint(-2, 1, -1, 0)) == "(-2+i-j+0k)"
        assert repr(hurwitzint(10**20, -1, 0, 3)) == f"({10**20}-i+0j+3k)"
