from __future__ import annotations

import copy
import operator
import os
import pickle
import random

from fractions import Fraction
from itertools import product, starmap
from math import floor, gcd, isqrt
from pathlib import Path

import pytest

from hurwitz import HurwitzQuaternion

import quatint.quat

from quatint.quat import NonCommutativeFactorization, hurwitzint, prod_left, prod_right, rdivmod

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
    def wide_search_values():
        """Every Lipschitz integer with components in [-3, 3], and every half-integer with numerators in [-5, 5]"""
        for a, b, c, d in product(range(-3, 4), repeat=4):
            yield hurwitzint(a, b, c, d)

        for a, b, c, d in product(range(-5, 6, 2), repeat=4):
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
    def smallest_remainder_norm(a: hurwitzint, b: hurwitzint, *, right: bool = False) -> int:
        """
        Brute-force the smallest N(r) that any Hurwitz quotient q can leave, in a = q*b + r (or a = b*q + r).

        In numerator units the exact quotient is U/n, where n = N(b) and U = a*conj(b) (or conj(b)*a). The nearest
            Hurwitz integer has, in every part, one of the two even integers around U_i/n, or else one of the two odd
            ones around it, so trying all 2 * 2**4 of those is sure to include it.

        Returns:
            int: The smallest possible remainder norm.
        """
        n = abs(b)
        U = list(b.conjugate() * a) if right else list(a * b.conjugate())

        smallest = None
        for parity in (0, 1):
            # The largest integer of this parity that is <= U_i/n, and so the one 2 above it is > U_i/n
            lows = [u // n - ((u // n - parity) & 1) for u in U]
            for s0, s1, s2, s3 in product((0, 2), repeat=4):
                q = hurwitzint(lows[0] + s0, lows[1] + s1, lows[2] + s2, lows[3] + s3, half=True)
                r = a - b * q if right else a - q * b
                if smallest is None or abs(r) < smallest:
                    smallest = abs(r)

        assert smallest is not None
        return smallest


class TestEq(HurwitzIntTests):
    """Tests for __eq__"""

    def test_main(self):
        """Basic equals tests"""
        c = hurwitzint(1, 2, 3, 4)
        assert self.a_int == c
        assert self.b_int != c

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


class TestImmutable(HurwitzIntTests):
    """Tests that a hurwitzint cannot be changed once it is made"""

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

    def test_copy_and_pickle(self):
        """Copies and pickled round trips are still equal to the original"""
        for x in (hurwitzint(1, 2, 3, 4), hurwitzint(3, -5, 7, 9, half=True)):
            assert copy.copy(x) == x
            assert copy.deepcopy(x) == x
            assert pickle.loads(pickle.dumps(x)) == x


class TestComponents(HurwitzIntTests):
    """Tests for reading a hurwitzint's parts: a, b, c and d, len, iteration, indexing, den and is_lipschitz"""

    def test_parts_are_numerators(self):
        """However they are read, the parts are the numerators over 2, and they rebuild the value with half=True"""
        for x, numerators in ((hurwitzint(1, -2, 3, 0), [2, -4, 6, 0]),
                              (hurwitzint(3, -5, 7, 9, half=True), [3, -5, 7, 9])):
            assert len(x) == 4
            assert list(x) == numerators
            assert [x[0], x[1], x[2], x[3]] == numerators
            assert [x.a, x.b, x.c, x.d] == numerators
            assert hurwitzint(*x, half=True) == x

    def test_index_out_of_range(self):
        """Indexing past the four parts raises IndexError"""
        x = hurwitzint(1, 2, 3, 4)
        for idx in (4, 5, 100):
            with pytest.raises(IndexError):
                operator.getitem(x, idx)

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
        for x in (hurwitzint(2), hurwitzint(1, 1, 0, 0), hurwitzint(3, 5, 7, 9, half=True)):
            with pytest.raises(ValueError, match="Negative powers"):
                pow(x, -1)


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

    def test_random_remainder_is_smallest(self):
        """For random pairs of every size, the remainder is as small as any Hurwitz quotient could make it"""
        rng = random.Random(1_000)
        for bound_a, bound_b in ((10, 3), (10, 10**4), (10**4, 10**2), (10**12, 10**5), (10**30, 10**12)):
            for _ in range(150):
                a = self.rand_hurwitzint(rng, bound_a)
                b = self.rand_hurwitzint(rng, bound_b)
                q, r = divmod(a, b)

                assert q * b + r == a
                assert 2 * abs(r) <= abs(b)
                assert abs(r) == self.smallest_remainder_norm(a, b)

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

    def test_random_remainder_is_smallest(self):
        """The right-division version of TestDiv.test_random_remainder_is_smallest"""
        rng = random.Random(3_000)
        for bound_a, bound_b in ((10, 3), (10, 10**4), (10**4, 10**2), (10**12, 10**5), (10**30, 10**12)):
            for _ in range(150):
                a = self.rand_hurwitzint(rng, bound_a)
                b = self.rand_hurwitzint(rng, bound_b)
                q, r = rdivmod(a, b)

                assert b * q + r == a
                assert 2 * abs(r) <= abs(b)
                assert abs(r) == self.smallest_remainder_norm(a, b, right=True)

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


class TestRoundDivTiesAwayFromZero:
    """Tests for _round_div_ties_away_from_zero, the rounding that every division starts from"""

    def test_matches_exact_rounding(self):
        """a/b rounds to the nearest integer, with exact halves going away from zero, for small and big values"""
        values = (*range(-60, 61), 10**30 + 5, -(10**30) - 5, 2**64 + 1, -(2**64) - 1)
        for b in (*range(1, 13), 2**64):
            for a in values:
                x = Fraction(a, b)
                nearest = floor(abs(x) + Fraction(1, 2))

                assert quatint.quat._round_div_ties_away_from_zero(a, b) == (nearest if x >= 0 else -nearest)

    def test_divisor_must_be_positive(self):
        """The divisor b has to be positive (it is always a norm), and anything else raises ValueError"""
        for b in (0, -1, -7):
            with pytest.raises(ValueError, match="b must be > 0"):
                quatint.quat._round_div_ties_away_from_zero(5, b)


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

    def assert_factoring(self, n: hurwitzint, factors: NonCommutativeFactorization):
        """Validate everything about the factoring is correct"""
        ans = factors.prod_right()

        self.assert_equal(n, ans)

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

    def assert_factoring(self, n: hurwitzint, factors: NonCommutativeFactorization):
        """Validate everything about the factoring is correct"""
        ans = factors.prod_left()

        self.assert_equal(n, ans)

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
