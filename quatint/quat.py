from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import gcd, prod
from typing import ClassVar, Iterable, Iterator, Literal, Union

from sympy import factorint, isprime

OTHER_OP_TYPES = int | float
_OTHER_OP_TYPES = (int, float)  # mypyc-friendly for isinstance
OP_TYPES = Union["hurwitzint", OTHER_OP_TYPES]


@dataclass(frozen=True, slots=True)
class NonCommutativeFactorization:
    """
    Normal form of a factorization into Hurwitz primes:

        direction="right":  x = content * unit * P1 * P2 * ... * Pk
        direction="left":   x = content * Pk * ... * P2 * P1 * unit

    - content is the integer part of x left unfactored: from factor_right_detail or factor_left_detail, the largest
        positive integer dividing x (0 for x = 0). That integer is unique, but its factorization into Hurwitz primes
        is not, so it is left to expand_content, which makes a fixed choice, moves those primes in with the others,
        and leaves content 1.
    - unit is a Hurwitz unit (norm 1).
    - Pi are Hurwitz primes (norm is a rational prime) sorted by norm, smallest first,
        each normalized by unit-migration.
    """
    content: int
    unit: hurwitzint
    primes: tuple[hurwitzint, ...]
    direction: Literal["left", "right"]

    def expand_content(self, factors: dict[int, int] | None = None) -> NonCommutativeFactorization:
        """
        Return this factorization with the content factored into Hurwitz primes too, so that content becomes 1.

        Every rational prime p is conj(P) * P for any Hurwitz prime P of norm p, but there is no unique way to pick P:
            p + 1 of them (just 1 for p=2) are not associates of each other, and each gives a different factorization
            (Conway & Smith call this recombination). So this makes a fixed choice,
            P = hurwitzint.prime_of_norm(p, direction=self.direction), for every p.

        Each p of the content goes in as the pair conj(P), P, ahead of any other primes of norm p. Then every factor
            is made canonical again, working from the far end toward the unit: each factor takes in the unit left
            over from the one before, and passes on what it has to give up to be canonical, until the last one
            joins `unit`. That is unit migration, so the product never changes, and the result keeps every guarantee
            of the primes from factor_right_detail and factor_left_detail: sorted by norm, each canonical, and
            a unit on the same side of x as `unit` only changes `unit`.

        This has to factor the content as an integer, which factor_right_detail and factor_left_detail never do,
            and that is slow for a big content with big prime factors. If they are known, pass them as factors
            ({prime: exponent}, like sympy.factorint returns) to skip it.

        Args:
            factors: The prime factorization of content, if it is already known.

        Returns:
            NonCommutativeFactorization: The expanded factorization, or this one if content is 0 (which has no
                factorization) or 1 (which leaves nothing to expand).

        Raises:
            TypeError: If factors has anything but ints in it.
            ValueError: If factors does not multiply to content, or has a key that is not prime.
        """
        m = self.content
        if factors is not None:
            # The mypyc build rejects anything but ints before getting here, so this makes pure Python match it
            if not all(isinstance(p, int) and isinstance(e, int) for p, e in factors.items()):
                raise TypeError("factors must map int primes to int exponents")

            # prime_of_norm checks that each p is prime, below
            if prod(p**e for p, e in factors.items()) != m:
                raise ValueError(f"factors must multiply to the content, {m}")

        if m <= 1:
            return self

        nf = _factorint(m) if factors is None else factors

        # Each p of the content goes in as conj(P), P, which is p whichever way it is multiplied out:
        #   conj(P) * P in a right factorization, and P * conj(P) in a left one (whose product runs right to left)
        primes = list(self.primes)
        norms = [abs(prime) for prime in primes]
        for p in sorted(nf):
            P = hurwitzint.prime_of_norm(p, direction=self.direction)
            i = bisect_left(norms, p)
            primes[i:i] = [P.conjugate(), P] * nf[p]
            norms[i:i] = [p] * (2 * nf[p])

        # Make every factor canonical again, from the far end toward the unit, carrying the leftover units along
        right = self.direction == "right"
        carry = hurwitzint(1)
        for i in range(len(primes) - 1, -1, -1):
            if right:
                # canon == u * (prime * carry), so prime * carry == conj(u) * canon, and conj(u) moves on to the left
                canon, u = (primes[i] * carry)._canonical_associate("left")
            else:
                # canon == (carry * prime) * u, so carry * prime == canon * conj(u), and conj(u) moves on to the right
                canon, u = (carry * primes[i])._canonical_associate("right")

            primes[i] = canon
            carry = u.conjugate()  # A unit's inverse is its conjugate

        unit = self.unit * carry if right else carry * self.unit
        return NonCommutativeFactorization(content=1, unit=unit, primes=tuple(primes), direction=self.direction)

    def __reduce__(self) -> tuple:
        # mypyc's default pickling would set the fields one at a time, which a frozen dataclass refuses
        return NonCommutativeFactorization, (self.content, self.unit, self.primes, self.direction)

    def prod(self):
        """Recreate the number using prod_right or prod_left"""
        if self.direction == "right":
            return self.prod_right()
        return self.prod_left()

    def prod_right(self):
        """Recreate the number using prod_right"""
        return prod_right(self.primes, start=self.unit * self.content)

    def prod_left(self):
        """Recreate the number using prod_left"""
        return prod_left(self.primes, start=self.unit * self.content)


def _mul(x: int, y: int) -> int:
    """
    Return x * y, worked out from the magnitudes of x and y.

    This only matters for speed under mypyc. Its int multiplication is a plain machine multiply when both operands
        are non-negative and below 2**30, but for anything else it boxes both ints and calls Python's own
        multiplication (see CPyTagged_IsMultiplyOverflow in mypyc's lib-rt/CPy.h). The parts of a quaternion are
        negative about half the time, so the hot paths multiply through this, which keeps products of small values on
        the fast path whatever their signs. (In pure Python it is slower than a plain x * y, since each product
        becomes a function call.)

    Returns:
        int: x * y.
    """
    if x < 0:
        if y < 0:
            return (-x) * (-y)
        return -((-x) * y)
    if y < 0:
        return -(x * (-y))
    return x * y


def _round_div_ties_away_from_zero(a: int, b: int) -> int:
    """Round a/b to nearest integer; ties go away from zero. b must be > 0."""
    if b <= 0:
        raise ValueError("b must be > 0")

    if a >= 0:
        return (a + (b // 2)) // b

    # a < 0
    return -((-a + (b // 2)) // b)


def _nearest_with_parity(U: int, Q0: int, parity: int, n: int) -> tuple[int, int]:
    """
    Return (Q, metric) minimizing (Q*n - U)^2 subject to Q % 2 == parity,
    given Q0, the nearest integer to U/n.

    Returns:
        tuple: The (Q, metric).
    """
    if parity == (Q0 & 1):
        d = _mul(Q0, n) - U
        return Q0, _mul(d, d)

    # Nearest integer with opposite parity must be Q0-1 or Q0+1.
    Qm = Q0 - 1
    Qp = Q0 + 1
    dm = _mul(Qm, n) - U
    dp = _mul(Qp, n) - U
    mm = _mul(dm, dm)
    mp = _mul(dp, dp)

    # Deterministic tie-break: prefer the smaller Q (Qm) on equal metric.
    if mm <= mp:
        return Qm, mm
    return Qp, mp


def _mul_numerators(A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> tuple[int, int, int, int]:
    """
    Multiply two Hurwitz integers given as numerators, (A+Bi+Cj+Dk)/2 * (E+Fi+Gj+Hk)/2, like hurwitzint.__mul__.

    Working on plain ints means no hurwitzint gets built (or parity-checked) for an intermediate product.
        The product of two Hurwitz integers is one too, so each numerator over 4 is even and halves exactly.

    Returns:
        tuple: The product's numerators, over 2.
    """
    return ((_mul(A, E) - _mul(B, F) - _mul(C, G) - _mul(D, H)) // 2,
            (_mul(A, F) + _mul(B, E) + _mul(C, H) - _mul(D, G)) // 2,
            (_mul(A, G) - _mul(B, H) + _mul(C, E) + _mul(D, F)) // 2,
            (_mul(A, H) + _mul(B, G) - _mul(C, F) + _mul(D, E)) // 2)


def _divmod_numerators(A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int, n: int, *, right: bool) \
        -> tuple[int, int, int, int, int, int, int, int]:
    """
    Nearest-lattice division of x = (A+Bi+Cj+Dk)/2 by y = (E+Fi+Gj+Hk)/2 (with n = N(y) > 0), all as numerators.

    Chooses q in the Hurwitz parity lattice (all components same parity) minimizing
        sum_i (Qi*n - Ui)^2
    where U holds the numerators of x * conj(y). Since x / y = x * conj(y) / N(y), each U_i / n is a numerator
    of the exact quotient. (For right-division U holds conj(y) * x instead, for y^-1 * x.)

    This can be done by comparing only 2 candidates: the best all-even q vs best all-odd q.

    This is the hot path under division, every gcd and every factorization step, so it works on plain ints: under
        mypyc a hurwitzint for each intermediate value costs more than the arithmetic does. For the same reason it
        avoids closures, lambdas and star-args, which mypyc compiles to slow generic Python calls.

    Returns:
        tuple: The numerators of the quotient q, then those of the remainder, x - q*y (or x - y*q if right).
    """
    # Conjugating the divisor just flips the signs of its i, j and k parts
    if right:
        Ua, Ub, Uc, Ud = _mul_numerators(E, -F, -G, -H, A, B, C, D)
    else:
        Ua, Ub, Uc, Ud = _mul_numerators(A, B, C, D, E, -F, -G, -H)

    # Unconstrained nearest integers to U_i / n (ties away from zero).
    A0 = _round_div_ties_away_from_zero(Ua, n)
    B0 = _round_div_ties_away_from_zero(Ub, n)
    C0 = _round_div_ties_away_from_zero(Uc, n)
    D0 = _round_div_ties_away_from_zero(Ud, n)

    if (((A0 ^ B0) & 1) == 0) and (((A0 ^ C0) & 1) == 0) and (((A0 ^ D0) & 1) == 0):
        # Fast path: already in the Hurwitz parity lattice.
        Qa, Qb, Qc, Qd = A0, B0, C0, D0
    else:
        # Compare best all-even vs best all-odd. The metric is a sum over the components,
        #   so each candidate is just the nearest integer of its parity, one component at a time.
        Ae, mAe = _nearest_with_parity(Ua, A0, 0, n)
        Be, mBe = _nearest_with_parity(Ub, B0, 0, n)
        Ce, mCe = _nearest_with_parity(Uc, C0, 0, n)
        De, mDe = _nearest_with_parity(Ud, D0, 0, n)

        Ao, mAo = _nearest_with_parity(Ua, A0, 1, n)
        Bo, mBo = _nearest_with_parity(Ub, B0, 1, n)
        Co, mCo = _nearest_with_parity(Uc, C0, 1, n)
        Do, mDo = _nearest_with_parity(Ud, D0, 1, n)

        # Deterministic tie-break if equal metric: prefer even.
        if mAe + mBe + mCe + mDe <= mAo + mBo + mCo + mDo:
            Qa, Qb, Qc, Qd = Ae, Be, Ce, De
        else:
            Qa, Qb, Qc, Qd = Ao, Bo, Co, Do

    # The remainder is x - q*y (or x - y*q)
    if right:
        Pa, Pb, Pc, Pd = _mul_numerators(E, F, G, H, Qa, Qb, Qc, Qd)
    else:
        Pa, Pb, Pc, Pd = _mul_numerators(Qa, Qb, Qc, Qd, E, F, G, H)

    return Qa, Qb, Qc, Qd, A - Pa, B - Pb, C - Pc, D - Pd


def _sub_mul_numerators(x: tuple[int, int, int, int],
                        q: tuple[int, int, int, int],
                        y: tuple[int, int, int, int],
                        *,
                        right: bool) -> tuple[int, int, int, int]:
    """
    Return x - q*y (or x - y*q if right), with each Hurwitz integer given as its numerators, like _mul_numerators.

    This steps the Bezout coefficients of the extended gcd along with its remainders (see hurwitzint._xgcd).

    Returns:
        tuple: The numerators of the result, over 2.
    """
    xa, xb, xc, xd = x
    Qa, Qb, Qc, Qd = q
    ya, yb, yc, yd = y
    if right:
        Pa, Pb, Pc, Pd = _mul_numerators(ya, yb, yc, yd, Qa, Qb, Qc, Qd)
    else:
        Pa, Pb, Pc, Pd = _mul_numerators(Qa, Qb, Qc, Qd, ya, yb, yc, yd)

    return xa - Pa, xb - Pb, xc - Pc, xd - Pd


def _uv_for_prime(p: int) -> tuple[int, int]:
    """
    Return the least u >= 0 for which -1 - u^2 is a square mod the prime p, and v, the least square root of it.

    Then 1 + u^2 + v^2 == 0 (mod p). This only needs pow (no library call that could change between versions),
        so the pair depends on nothing but p, and so does every prime built from it (see hurwitzint.prime_of_norm).

    Returns:
        tuple: (u, v).
    """
    if p == 2:
        return 0, 1

    if p % 4 == 1:
        # -1 is a square, so u = 0, and z^((p-1)/4) is a square root of -1 for any non-residue z
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1

        r = pow(z, (p - 1) // 4, p)
        return 0, min(r, p - r)

    # p % 4 == 3, so -1 is not a square, but -1 - u^2 is for about half of all u, and then t^((p+1)/4) is a root of t
    u = 1
    while True:
        t = (-1 - u * u) % p
        if pow(t, (p - 1) // 2, p) == 1:
            r = pow(t, (p + 1) // 4, p)
            return u, min(r, p - r)

        u += 1


def _factorint(n: int) -> dict[int, int]:
    """
    Return sympy.factorint(n), with every prime and exponent a plain int.

    With gmpy2 or python-flint installed, sympy 1.13 and later can return their own integer types (mpz or fmpz) for
        some factors, depending on which of its algorithms found them: the first factorint(7 * 104729**3) comes back
        as {7: 1, mpz(104729): mpz(3)}. Those are not ints, so the mypyc build would raise TypeError wherever one
        reached a parameter typed int.

    Returns:
        dict: {prime: exponent}, like sympy.factorint.
    """
    return {int(p): int(e) for p, e in factorint(n).items()}


def _from_numerators(A: int, B: int, C: int, D: int) -> hurwitzint:
    """
    Return the hurwitzint (A + B*i + C*j + D*k) / 2, which is how an unpickled one is rebuilt (see __reduce__).

    Pickles refer to this function by name, so renaming or removing it would break every pickle made before.

    Returns:
        hurwitzint: The value.
    """
    return hurwitzint(A, B, C, D, half=True)


class hurwitzint:
    """
    Hurwitz quaternion integer.

    Internally stored in "numerator units" as (A, B, C, D) representing:
        (A + B*i + C*j + D*k) / 2

    Integrality constraint (Hurwitz order):
        A, B, C, D must all have the same parity
        (all even = Lipschitz, all odd = true Hurwitz half-integer element).

    Notes:
      - Multiplication is non-commutative.
      - The reduced norm is always an integer for valid Hurwitz elements:
            N(q) = (A^2 + B^2 + C^2 + D^2) / 4
      - Values are immutable, like int or Fraction: a, b, c and d are read-only views of the numerators,
            so a hurwitzint is safe to hash, and to use in sets and as a dict key.
    """

    __slots__ = ("_a", "_b", "_c", "_d")

    _a: int
    _b: int
    _c: int
    _d: int

    UNITS: ClassVar[list[hurwitzint]] = []

    def __init__(
        self,
        a: OTHER_OP_TYPES = 0,
        b: OTHER_OP_TYPES = 0,
        c: OTHER_OP_TYPES = 0,
        d: OTHER_OP_TYPES = 0,
        *,
        half: bool = False,
    ) -> None:
        """
        Initialize a hurwitzint.

        Args:
            a:
                If half=False (default): interpreted as integer components (Lipschitz):
                    q = a + b*i + c*j + d*k
                If half=True: interpreted as numerator components for /2:
                    q = (a + b*i + c*j + d*k) / 2
                (So (1+i+j+k)/2 is hurwitzint(1,1,1,1, half=True).)
                Floats are truncated with int(), like everywhere else a float meets a hurwitzint.
            b: See a.
            c: See a.
            d: See a.
            half:
                Whether inputs are already in numerator-units for /2 representation.

        Raises:
            ValueError: If parity is incorrect.
        """
        # This runs for the result of every operation (see _make), and under mypyc int() is slow on an argument
        #   that may also be a float, so plain ints skip it (type() rather than isinstance(), so bools become ints)
        a0 = a if type(a) is int else int(a)
        b0 = b if type(b) is int else int(b)
        c0 = c if type(c) is int else int(c)
        d0 = d if type(d) is int else int(d)

        if not half:
            a0 *= 2
            b0 *= 2
            c0 *= 2
            d0 *= 2

        # All four must have the same parity.
        if ((a0 ^ b0) & 1) or ((a0 ^ c0) & 1) or ((a0 ^ d0) & 1):
            raise ValueError("For Hurwitz integers, a,b,c,d must all have the same parity")

        self._a, self._b, self._c, self._d = a0, b0, c0, d0

    @property
    def a(self) -> int:
        """Numerator of the real part (which is a / 2)"""
        return self._a

    @property
    def b(self) -> int:
        """Numerator of the i part (which is b / 2)"""
        return self._b

    @property
    def c(self) -> int:
        """Numerator of the j part (which is c / 2)"""
        return self._c

    @property
    def d(self) -> int:
        """Numerator of the k part (which is d / 2)"""
        return self._d

    # region constructors / conversions
    @classmethod
    def _make(cls, A: int, B: int, C: int, D: int) -> hurwitzint:
        """Construct a new value of *this* conceptual type from internal numerators A,B,C,D."""
        return cls(A, B, C, D, half=True)

    @classmethod
    def _from_obj(cls, n: OP_TYPES) -> hurwitzint:
        """Convert a random object to a hurwitzint"""
        if isinstance(n, _OTHER_OP_TYPES):
            # scalar n -> (2n + 0i + 0j + 0k)/2
            return cls._make(2 * int(n), 0, 0, 0)

        if isinstance(n, hurwitzint):
            return n

        return NotImplemented
    # endregion

    @property
    def is_lipschitz(self) -> bool:
        """True iff all components are integers (i.e., all numerators even)."""
        return ((self._a | self._b | self._c | self._d) & 1) == 0

    @property
    def den(self) -> int:
        """Static denominator, because everything is doubled under the hood anyway"""
        return 2

    def conjugate(self) -> hurwitzint:
        """Quaternion conjugation: a+bi+cj+dk -> a-bi-cj-dk (in numerator units)."""
        return self._make(self._a, -self._b, -self._c, -self._d)

    def __add__(self, other: OP_TYPES) -> hurwitzint:
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, hurwitzint):
            return self._make(self._a + other._a, self._b + other._b, self._c + other._c, self._d + other._d)

        return NotImplemented

    def __radd__(self, other: OTHER_OP_TYPES) -> hurwitzint:
        return self.__add__(other)

    def __sub__(self, other: OP_TYPES) -> hurwitzint:
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, hurwitzint):
            return self._make(self._a - other._a, self._b - other._b, self._c - other._c, self._d - other._d)

        return NotImplemented

    def __rsub__(self, other: OTHER_OP_TYPES) -> hurwitzint:
        return self.__neg__().__add__(other)

    def __neg__(self) -> hurwitzint:
        return self._make(-self._a, -self._b, -self._c, -self._d)

    def __pos__(self) -> hurwitzint:
        return self._make(self._a, self._b, self._c, self._d)

    def __mul__(self, other: OP_TYPES) -> hurwitzint:
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            return NotImplemented

        # Quaternion multiplication in numerator units.
        # If q=(A+Bi+Cj+Dk)/2 and r=(E+Fi+Gj+Hk)/2,
        # then qr has denominator 4; we store with denominator 2,
        # so we must divide resulting numerators by 2.
        A, B, C, D = self._a, self._b, self._c, self._d
        E, F, G, H = other._a, other._b, other._c, other._d

        # (a,b,c,d)*(e,f,g,h) with i^2=j^2=k^2=ijk=-1:
        P = _mul(A, E) - _mul(B, F) - _mul(C, G) - _mul(D, H)
        Q = _mul(A, F) + _mul(B, E) + _mul(C, H) - _mul(D, G)
        R = _mul(A, G) - _mul(B, H) + _mul(C, E) + _mul(D, F)
        S = _mul(A, H) + _mul(B, G) - _mul(C, F) + _mul(D, E)

        # Must be divisible by 2 to land back in the Hurwitz order.
        if (P & 1) or (Q & 1) or (R & 1) or (S & 1):
            raise ArithmeticError("Non-integral product; parity constraint violated")

        return self._make(P // 2, Q // 2, R // 2, S // 2)

    def __rmul__(self, other: OTHER_OP_TYPES) -> hurwitzint:
        return self.__mul__(other)

    # Not just `exp: float`, since mypyc would turn an int exponent into a double and lose every bit past 2**53
    def __pow__(self, exp: OTHER_OP_TYPES) -> hurwitzint:
        # The mypyc build rejects anything else before getting here, so this makes pure Python match it
        if not isinstance(exp, _OTHER_OP_TYPES):
            return NotImplemented

        e = int(exp)
        if e < 0:
            raise ValueError("Negative powers not supported")

        result = hurwitzint(1, 0, 0, 0)  # multiplicative identity
        base: hurwitzint = self
        while e:
            if e & 1:
                result *= base

            e >>= 1
            if e:
                base *= base

        return result

    # region Euclidean division (Hurwitz order is norm-Euclidean)
    def _division(self,
                  divisor: hurwitzint,
                  divisor_norm: int,
                  *,
                  right: bool = False) -> tuple[hurwitzint, hurwitzint]:
        """
        The division shared by __divmod__ and rdivmod: the nearest-lattice division of _divmod_numerators.

        Args:
            divisor: The divisor.
            divisor_norm: The divisor norm. Should be checked for 0 already!
            right: Divide on the right (self = divisor*q + r), rather than on the left (self = q*divisor + r).

        Returns:
            tuple: The quotient and remainder.
        """
        Qa, Qb, Qc, Qd, Ra, Rb, Rc, Rd = _divmod_numerators(self._a, self._b, self._c, self._d,
                                                            divisor._a, divisor._b, divisor._c, divisor._d,
                                                            divisor_norm, right=right)
        return self._make(Qa, Qb, Qc, Qd), self._make(Ra, Rb, Rc, Rd)

    # region Left-division helpers (non-commutative!)
    def __divmod__(self, other: OP_TYPES) -> tuple[hurwitzint, hurwitzint]:
        """
        Nearest-lattice division in the Hurwitz quaternion order.

        We define quotient on the LEFT (Python-style):
            self = q * other + r

        Because multiplication is non-commutative, this is a specific choice.

        Returns:
            (q, r) where r has small norm (typically < abs(other)),
                or NotImplemented if other is an unsupported type.

        Raises:
            ZeroDivisionError: if other == 0
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            return NotImplemented

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        # q ~ self * conj(other) / N(other), the usual quaternion self / other (see _division)
        return self._division(other, n)

    def __rdivmod__(self, other: OTHER_OP_TYPES) -> tuple[hurwitzint, hurwitzint]:
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__divmod__(self)

        return NotImplemented

    def __truediv__(self, other: OP_TYPES) -> hurwitzint:
        # mirror QuadInt: treat / as Euclidean division in this domain
        return self.__floordiv__(other)

    def __rtruediv__(self, other: OTHER_OP_TYPES) -> hurwitzint:
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__truediv__(self)

        return NotImplemented

    def __floordiv__(self, other: OP_TYPES) -> hurwitzint:
        # Not divmod(self, other), so an unsupported other gets its own __rfloordiv__ and a TypeError naming //
        qr = self.__divmod__(other)
        if qr is NotImplemented:
            return NotImplemented

        return qr[0]

    def __rfloordiv__(self, other: OTHER_OP_TYPES) -> hurwitzint:
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__floordiv__(self)

        return NotImplemented

    def __mod__(self, other: OP_TYPES) -> hurwitzint:
        qr = self.__divmod__(other)
        if qr is NotImplemented:
            return NotImplemented

        return qr[1]

    def __rmod__(self, other: OTHER_OP_TYPES) -> hurwitzint:
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__mod__(self)

        return NotImplemented
    # endregion

    # region Right-division helpers
    def rdivmod(self, other: OP_TYPES) -> tuple[hurwitzint, hurwitzint]:
        """
        Right-quotient division in the Hurwitz quaternion order.

        Defines quotient on the RIGHT:
            self = other * q + r

        Returns:
            (q, r)

        Raises:
            TypeError: If other is an unsupported type.
            ZeroDivisionError: If trying to divide by 0.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for rdivmod: {type(other).__name__!r}")

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        # Right quotient: q ~ other^{-1} * self = conj(other) * self / N(other) (see _division)
        return self._division(other, n, right=True)

    def rtruediv(self, other: OP_TYPES) -> hurwitzint:
        """A version of __truediv__ for right-division"""
        return self.rfloordiv(other)

    def rfloordiv(self, other: OP_TYPES) -> hurwitzint:
        """A version of __floordiv__ for right-division"""
        q, _ = self.rdivmod(other)
        return q

    def rmod(self, other: OP_TYPES) -> hurwitzint:
        """A version of __mod__ for right-division"""
        _, r = self.rdivmod(other)
        return r
    # endregion
    # endregion

    # region Exact division
    def _exact_division(self,
                        divisor: hurwitzint,
                        divisor_norm: int,
                        *,
                        right: bool = False) -> hurwitzint | None:
        """
        The exact division shared by exact_div_right and exact_div_left.

        The quotient self / divisor is self * conj(divisor) / N(divisor) (or conj(divisor) * self / N(divisor)
            with right=True), as in _divmod_numerators, so each numerator of self * conj(divisor), over N(divisor),
            is a numerator of the quotient. That is a Hurwitz integer only if all four divide exactly,
            into four numerators of the same parity.

        Args:
            divisor: The divisor.
            divisor_norm: The divisor norm. Should be checked for 0 already!
            right: Divide on the right (self == divisor*q), rather than on the left (self == q*divisor).

        Returns:
            hurwitzint | None: The quotient q, or None if no Hurwitz integer q divides exactly.
        """
        A, B, C, D = self._a, self._b, self._c, self._d
        E, F, G, H = divisor._a, divisor._b, divisor._c, divisor._d
        n = divisor_norm

        # Conjugating the divisor just flips the signs of its i, j and k parts
        if right:
            Ua, Ub, Uc, Ud = _mul_numerators(E, -F, -G, -H, A, B, C, D)
        else:
            Ua, Ub, Uc, Ud = _mul_numerators(A, B, C, D, E, -F, -G, -H)

        if Ua % n or Ub % n or Uc % n or Ud % n:
            return None

        Qa, Qb, Qc, Qd = Ua // n, Ub // n, Uc // n, Ud // n

        # Integer numerators of mixed parity are a quotient outside the Hurwitz order, like 1 / (1+i) == (1-i)/2
        if ((Qa ^ Qb) & 1) or ((Qa ^ Qc) & 1) or ((Qa ^ Qd) & 1):
            return None

        return self._make(Qa, Qb, Qc, Qd)

    def exact_div_right(self, other: OP_TYPES) -> hurwitzint | None:
        """
        Return q if self == q * other exactly, else None.

        This divides other off the right of self, when other is a right divisor of self: like gcd_right,
            it is named for the side of self that other divides. It is the exact version of divmod(self, other),
            whose quotient it returns whenever the remainder is 0, and it returns None otherwise.

        Returns:
            hurwitzint | None: The quotient q, or None if no Hurwitz integer q gives self == q * other.

        Raises:
            TypeError: If other is an unsupported type.
            ZeroDivisionError: If other is 0.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for exact_div_right: {type(other).__name__!r}")

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        return self._exact_division(other, n)

    def exact_div_left(self, other: OP_TYPES) -> hurwitzint | None:
        """
        Return q if self == other * q exactly, else None.

        This divides other off the left of self, when other is a left divisor of self: like gcd_left,
            it is named for the side of self that other divides. It is the exact version of self.rdivmod(other),
            whose quotient it returns whenever the remainder is 0, and it returns None otherwise.

        Returns:
            hurwitzint | None: The quotient q, or None if no Hurwitz integer q gives self == other * q.

        Raises:
            TypeError: If other is an unsupported type.
            ZeroDivisionError: If other is 0.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for exact_div_left: {type(other).__name__!r}")

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        return self._exact_division(other, n, right=True)

    def divides_right(self, other: OP_TYPES) -> bool:
        """
        Return True iff self is a right divisor of other: other == q * self for some Hurwitz integer q.

        That is, iff other.exact_div_right(self) finds a quotient. Note which way round that is: self is the
            divisor, as in "self divides other". Everything divides 0, and 0 divides only 0, so unlike
            exact_div_right this never raises ZeroDivisionError.

        Returns:
            bool: Whether self divides other on the right.

        Raises:
            TypeError: If other is an unsupported type.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for divides_right: {type(other).__name__!r}")

        n = abs(self)
        if n == 0:
            return not other  # 0 only divides 0

        return other._exact_division(self, n) is not None

    def divides_left(self, other: OP_TYPES) -> bool:
        """
        Return True iff self is a left divisor of other: other == self * q for some Hurwitz integer q.

        That is, iff other.exact_div_left(self) finds a quotient. Note which way round that is: self is the
            divisor, as in "self divides other". Everything divides 0, and 0 divides only 0, so unlike
            exact_div_left this never raises ZeroDivisionError.

        Returns:
            bool: Whether self divides other on the left.

        Raises:
            TypeError: If other is an unsupported type.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for divides_left: {type(other).__name__!r}")

        n = abs(self)
        if n == 0:
            return not other  # 0 only divides 0

        return other._exact_division(self, n, right=True) is not None
    # endregion

    def __abs__(self) -> int:
        """
        Reduced norm:
            N((A+Bi+Cj+Dk)/2) = (A^2+B^2+C^2+D^2)/4

        Always an integer for valid Hurwitz integers.

        Returns:
            int: The norm.

        Raises:
            ArithmeticError: If there is a non-integral norm due to parity violation.
        """
        num = _mul(self._a, self._a) + _mul(self._b, self._b) + _mul(self._c, self._c) + _mul(self._d, self._d)
        # q, r = divmod(num, 4)   # Below is ever so slightly faster it seems, and this is an important operation
        r = num & 3
        q = num >> 2

        if r != 0:
            raise ArithmeticError("Non-integral norm; parity constraint violated")

        return q

    def __bool__(self) -> bool:
        return (self._a | self._b | self._c | self._d) != 0

    def __iter__(self) -> Iterator[int]:
        return iter((self._a, self._b, self._c, self._d))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> int:
        if idx == 0:
            return self._a
        if idx == 1:
            return self._b
        if idx == 2:
            return self._c
        if idx == 3:
            return self._d
        raise IndexError("hurwitzint index out of range (valid: 0..3)")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, hurwitzint):
            return (self._a, self._b, self._c, self._d) == (other._a, other._b, other._c, other._d)

        # Python numbers compare exactly (unlike arithmetic, which truncates floats with int()), since anything equal
        #   has to hash the same too (see __hash__). A complex off the real axis is never equal.
        if isinstance(other, int):
            return self._b == 0 and self._c == 0 and self._d == 0 and self._a == 2 * other

        if isinstance(other, (float, complex)):
            real, imag = other.real, other.imag
            if imag or not real.is_integer():
                return False

            return self._b == 0 and self._c == 0 and self._d == 0 and self._a == 2 * int(real)

        return False

    def __hash__(self) -> int:
        # Equal objects must hash the same, so a real value hashes like the int it equals (see __eq__)
        if self._b == 0 and self._c == 0 and self._d == 0:
            return hash(self._a // 2)

        return hash((self._a, self._b, self._c, self._d))

    def __reduce__(self) -> tuple:
        # Pickle as the numerators. Without this, pickling only works at protocol 2 and up: at 0 and 1, copyreg can
        #   rebuild neither a class with __slots__ (pure Python) nor a mypyc native class
        return _from_numerators, (self._a, self._b, self._c, self._d)

    def __repr__(self) -> str:
        if self.is_lipschitz:
            # If all even, show integer components without "/2".
            ra, rb, rc, rd = self._a // 2, self._b // 2, self._c // 2, self._d // 2
            den = None
        else:
            # Otherwise show numerator form "(...)/2".
            ra, rb, rc, rd = self._a, self._b, self._c, self._d
            den = 2

        # Special-case: only time we omit parentheses is when den is None and
        # ra=rb=rc=0, so we're displaying a pure k-term (like Python's "2j").
        if den is None and ra == 0 and rb == 0 and rc == 0:
            if rd == 1:
                return "k"
            if rd == -1:
                return "-k"
            return f"{rd}k"

        def _imag_term(coeff: int, sym: str) -> str:
            sign = "+" if coeff >= 0 else "-"
            mag = -coeff if coeff < 0 else coeff
            mag_str = "" if mag == 1 else str(mag)  # 1i -> i
            return f"{sign}{mag_str}{sym}"

        core = f"({ra}{_imag_term(rb, 'i')}{_imag_term(rc, 'j')}{_imag_term(rd, 'k')})"
        return f"{core}/{den}" if den is not None else core

    @property
    def is_unit(self) -> bool:
        """Is this a unit Hurwitz integer?"""
        return abs(self) == 1

    def inverse(self) -> hurwitzint:
        """Find the inverse of the current hurwitzint (only applies to units)"""
        if not self.is_unit:
            raise ValueError("only Hurwitz units have inverses in the Hurwitz order")

        return self.conjugate()

    def split_lipschitz(self) -> tuple[hurwitzint, hurwitzint | None]:
        """
        Return (whole, half_unit) such that self == whole + half_unit.

        Returns:
            tuple: (whole, half_unit)
                If self is already Lipschitz/integer-valued, returns (self, None).
                    Otherwise half_unit is one of the 16 Hurwitz half-units.
        """
        if self.is_lipschitz:
            return self, None

        def sgn(n: int) -> int:
            return 1 if n > 0 else -1

        half_unit = self._make(
            sgn(self._a),
            sgn(self._b),
            sgn(self._c),
            sgn(self._d),
        )
        whole = self - half_unit

        return whole, half_unit

    # region GCD
    def _gcd(self,
             other: OP_TYPES,
             *,
             right: bool = False) -> hurwitzint:
        """
        GCD via Euclidean algorithm, as whichever associate the algorithm lands on.

        This divides on the left (a = q*b + r) for a right gcd, or with right=True on the right (a = b*q + r)
            for a left gcd. The loop runs on plain int numerators (see _divmod_numerators), so the only
            hurwitzint it builds is the answer.

        Returns:
            hurwitzint: The gcd.

        Raises:
            TypeError: If other is an unsupported type.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"Unable to divide hurwitzint and type {type(other)}")

        if not self:
            return other

        A, B, C, D = self._a, self._b, self._c, self._d
        E, F, G, H = other._a, other._b, other._c, other._d
        while E or F or G or H:
            n = (_mul(E, E) + _mul(F, F) + _mul(G, G) + _mul(H, H)) >> 2
            _, _, _, _, Ra, Rb, Rc, Rd = _divmod_numerators(A, B, C, D, E, F, G, H, n, right=right)
            A, B, C, D, E, F, G, H = E, F, G, H, Ra, Rb, Rc, Rd

        return self._make(A, B, C, D)

    def _xgcd(self,
              other: OP_TYPES,
              *,
              right: bool = False) -> tuple[hurwitzint, hurwitzint, hurwitzint]:
        """
        The extended form of _gcd: the same gcd g, with Bezout coefficients s and t.

        _gcd divides on the left (a = q*b + r) for a right gcd, so every remainder it goes through is s*self + t*other
            for some s and t: the first two are self (s=1, t=0) and other (s=0, t=1), and each one after is
            a - q*b for the two before it, so its coefficients are those of a, minus q times those of b.
            With right=True it divides on the right (a = b*q + r) for a left gcd, so every remainder is
            self*s + other*t, and q goes on the right of the coefficients.

        This runs the same divisions as _gcd, on plain int numerators, so it lands on the same associate.

        Returns:
            tuple: (g, s, t), with s*self + t*other == g, or with right=True, self*s + other*t == g.

        Raises:
            TypeError: If other is an unsupported type.
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for xgcd: {type(other).__name__!r}")

        A, B, C, D = self._a, self._b, self._c, self._d
        E, F, G, H = other._a, other._b, other._c, other._d

        # The coefficients of A..D and of E..H, as numerators
        s0, t0 = (2, 0, 0, 0), (0, 0, 0, 0)
        s1, t1 = (0, 0, 0, 0), (2, 0, 0, 0)
        while E or F or G or H:
            n = (_mul(E, E) + _mul(F, F) + _mul(G, G) + _mul(H, H)) >> 2
            Qa, Qb, Qc, Qd, Ra, Rb, Rc, Rd = _divmod_numerators(A, B, C, D, E, F, G, H, n, right=right)
            q = (Qa, Qb, Qc, Qd)

            A, B, C, D, E, F, G, H = E, F, G, H, Ra, Rb, Rc, Rd
            s0, s1 = s1, _sub_mul_numerators(s0, q, s1, right=right)
            t0, t1 = t1, _sub_mul_numerators(t0, q, t1, right=right)

        sa, sb, sc, sd = s0
        ta, tb, tc, td = t0
        return self._make(A, B, C, D), self._make(sa, sb, sc, sd), self._make(ta, tb, tc, td)

    def gcd_right(self,
                  other: OP_TYPES,
                  *,
                  normalize: bool = True) -> hurwitzint:
        """
        Right gcd via left-division Euclidean algorithm.

        The result g is a "right gcd":
            a = q*b + r
            a = a' * g and b = b' * g

        A right gcd is only unique up to a unit on its left (u*g), so by default this returns the canonical one
            of those (see `_canonical_associate`). That makes it independent of the argument order, and of any
            units on the left of either argument. normalize=False skips that, returning whichever associate
            the Euclidean algorithm lands on.

        Returns:
            hurwitzint: The gcd.
        """
        g = self._gcd(other)
        return g._canonical_associate(direction="left")[0] if normalize else g

    def gcd_left(self,
                 other: OP_TYPES,
                 *,
                 normalize: bool = True) -> hurwitzint:
        """
        Left gcd via RIGHT-division Euclidean algorithm.

        The result g is a "left gcd":
            a = b*q + r
            a = g * a' and b = g * b'

        A left gcd is only unique up to a unit on its right (g*u), so by default this returns the canonical one
            of those (see `_canonical_associate`). That makes it independent of the argument order, and of any
            units on the right of either argument. normalize=False skips that, returning whichever associate
            the Euclidean algorithm lands on.

        Returns:
            hurwitzint: The gcd.
        """
        g = self._gcd(other, right=True)
        return g._canonical_associate(direction="right")[0] if normalize else g

    def xgcd_right(self,
                   other: OP_TYPES,
                   *,
                   normalize: bool = True) -> tuple[hurwitzint, hurwitzint, hurwitzint]:
        """
        Extended right gcd: the gcd_right g of self and other, with Bezout coefficients s and t, on the left.

            s*self + t*other == g

        g is the same as self.gcd_right(other, normalize=normalize). Making it canonical multiplies it by a unit
            on its left, so s and t get the same unit on their left, which keeps the identity. With both self and
            other 0, this is (0, 1, 0).

        Returns:
            tuple: (g, s, t), with s*self + t*other == g.
        """
        g, s, t = self._xgcd(other)
        if normalize and g:
            g, u = g._canonical_associate(direction="left")  # g == u * (the g before)
            s, t = u * s, u * t

        return g, s, t

    def xgcd_left(self,
                  other: OP_TYPES,
                  *,
                  normalize: bool = True) -> tuple[hurwitzint, hurwitzint, hurwitzint]:
        """
        Extended left gcd: the gcd_left g of self and other, with Bezout coefficients s and t, on the right.

            self*s + other*t == g

        g is the same as self.gcd_left(other, normalize=normalize). Making it canonical multiplies it by a unit
            on its right, so s and t get the same unit on their right, which keeps the identity. With both self and
            other 0, this is (0, 1, 0).

        Returns:
            tuple: (g, s, t), with self*s + other*t == g.
        """
        g, s, t = self._xgcd(other, right=True)
        if normalize and g:
            g, u = g._canonical_associate(direction="right")  # g == (the g before) * u
            s, t = s * u, t * u

        return g, s, t
    # endregion

    # region Factoring
    def content(self) -> int:
        """
        Largest positive integer m such that self = m*q' with q' still a Hurwitz integer.
            Computed in numerator-units with the Hurwitz parity constraint.

        Returns:
            int: Computed content value.
        """
        A, B, C, D = self._a, self._b, self._c, self._d
        g = gcd(abs(A), abs(B), abs(C), abs(D))

        if g == 0:
            return 0

        # Adjust by powers of two until the reduced tuple is all same parity.
        while g > 0:
            a, b, c, d = A // g, B // g, C // g, D // g
            if (((a ^ b) & 1) == 0) and (((a ^ c) & 1) == 0) and (((a ^ d) & 1) == 0):
                return g
            g //= 2

        return 1  # practically unreachable for nonzero, but safe

    def _canonical_associate(self, direction: Literal["left", "right"]) -> tuple[hurwitzint, hurwitzint]:
        """
        Pick the canonical one of the 24 associates that differ from self by a unit on one side.

        direction="right" considers p*u, and direction="left" considers u*p (u a unit).
            The canonical one has the largest numerator tuple: the largest real part, with ties going to the
            largest i, then j, then k part. So for example a nonzero integer always comes out positive.

        The candidates are compared as plain int numerators (see _mul_numerators), so the only hurwitzint this
            builds is the answer. (The 24 associates of a nonzero value all differ, so the largest one is unique.)

        Returns:
             tuple: (p_canon, u) such that p_canon = p*u (direction="right") or p_canon = u*p (direction="left").
        """
        A, B, C, D = self._a, self._b, self._c, self._d
        right = direction == "right"

        # Every candidate's real part is at least -(|A| + |B| + |C| + |D|), so the first candidate always beats this
        Ba = -(abs(A) + abs(B) + abs(C) + abs(D)) - 1
        Bb = Bc = Bd = 0
        best_u = hurwitzint.UNITS[0]
        for u in hurwitzint.UNITS:
            E, F, G, H = u._a, u._b, u._c, u._d

            # The real part is the same for u*p as for p*u, and on its own it rules out most candidates,
            #   so the rest of the product is only worked out for a real part at least as large as the best one's
            real = (_mul(A, E) - _mul(B, F) - _mul(C, G) - _mul(D, H)) // 2
            if real < Ba:
                continue

            if right:
                Pa, Pb, Pc, Pd = _mul_numerators(A, B, C, D, E, F, G, H)
            else:
                Pa, Pb, Pc, Pd = _mul_numerators(E, F, G, H, A, B, C, D)

            # Compare the numerator tuples (Pa, Pb, Pc, Pd) > (Ba, Bb, Bc, Bd) one part at a time,
            #   since mypyc compiles a tuple comparison into a slow generic one
            if Pa != Ba:
                larger = Pa > Ba
            elif Pb != Bb:
                larger = Pb > Bb
            elif Pc != Bc:
                larger = Pc > Bc
            else:
                larger = Pd > Bd

            if larger:
                Ba, Bb, Bc, Bd = Pa, Pb, Pc, Pd
                best_u = u

        return self._make(Ba, Bb, Bc, Bd), best_u

    def _extract_right_prime(self, p: int) -> hurwitzint:
        """
        Extract a right prime of norm p dividing self.
            g = gcd_right(self, p)

        Returns:
            hurwitzint: The canonical Hurwitz prime with norm p that divides self on the right.

        Raises:
            ArithmeticError: If there is an unexpected problem preventing factoring, indicating a bug in the code.
        """
        # Directly extract from q via gcd with the scalar p
        g = self.gcd_right(p)

        if abs(g) != p:
            raise ArithmeticError(f"Failed to extract right prime for {p=}")

        return g

    def _extract_left_prime(self, p: int) -> hurwitzint:
        """
        Extract a left prime of norm p dividing self.
            g = gcd_left(self, p)

        Returns:
            hurwitzint: The canonical Hurwitz prime with norm p that divides self on the left.

        Raises:
            ArithmeticError: If there is an unexpected problem preventing factoring, indicating a bug in the code.
        """
        # Directly extract from q via gcd with the scalar p
        g = self.gcd_left(p)

        if abs(g) != p:
            raise ArithmeticError(f"Failed to extract left prime for {p=}")

        return g

    @classmethod
    def prime_of_norm(cls, p: OTHER_OP_TYPES, *, direction: Literal["left", "right"] = "right") -> hurwitzint:
        """
        Return a fixed Hurwitz prime whose norm is the rational prime p.

        Every rational prime is the norm of 24*(p + 1) Hurwitz integers (just 24 for p=2), so this picks one:
            with u the least u >= 0 for which -1 - u^2 is a square mod p, and v the least square root of it,
            p divides N(1 + u*i + v*j) = 1 + u^2 + v^2, and the canonical gcd of the two has norm p.
            direction="right" takes gcd_right, canonical up to a unit on its left like the primes of
            `factor_right_detail`, and direction="left" takes gcd_left, canonical up to a unit on its right like the
            primes of `factor_left_detail`.

        A quaternion times its conjugate is its norm, so this also factors p, as conj(P) * P (or P * conj(P)):

            hurwitzint.prime_of_norm(2) == 1+i, and 2 == (1-i) * (1+i)
            hurwitzint.prime_of_norm(5) == 2-j, and 5 == (2+j) * (2-j)

        Floats are truncated with int(), like everywhere else a float meets a hurwitzint.

        Returns:
            hurwitzint: The prime.

        Raises:
            TypeError: If p is not an int or float.
            ValueError: If p is not prime, or direction is not "left" or "right".
            ArithmeticError: If the gcd does not have norm p, indicating a bug in the code.
        """
        # The mypyc build rejects anything else before getting here, so this makes pure Python match it
        if not isinstance(p, _OTHER_OP_TYPES):
            raise TypeError(f"unsupported type for prime_of_norm: {type(p).__name__!r}")

        if direction not in ("left", "right"):
            raise ValueError(f"direction must be 'left' or 'right', not {direction!r}")

        # isprime also keeps composites away from _uv_for_prime, whose square root search might never end for one
        n = p if type(p) is int else int(p)
        if not isprime(n):
            raise ValueError(f"{n} is not prime")

        u, v = _uv_for_prime(n)
        seed = cls(1, u, v, 0)
        g = seed.gcd_right(n) if direction == "right" else seed.gcd_left(n)
        if abs(g) != n:
            raise ArithmeticError(f"Failed to find a prime of norm {n}")

        return g

    def factor_right_detail(self) -> NonCommutativeFactorization:
        """
        Deterministic right factorization normal form (primitive-first).

        Returns content, unit, and a tuple of Hurwitz primes P1..Pk, sorted by norm (smallest first), such that:
            self = content * unit * P1 * P2 * ... * Pk

        Primes are peeled off the right, largest norm first, so they land in sorted order without any swapping:
            a primitive quaternion has exactly one factorization for each ordering of its prime norms,
            up to unit migration (Conway & Smith). Each Pk is the canonical associate u*Pk of the right divisor
            with its norm, and the leftover unit migrates left, ending up in `unit`.

        Notes:
          - `content` stays an integer, since that is unique, and its factorization into Hurwitz primes is not
            (recombination). It also means the content never has to be factored as an integer, which can be slow.
            `expand_content()` on the result does both, making a fixed choice of primes.
          - Each Pi is irreducible because its norm is a rational prime.

        Returns:
            NonCommutativeFactorization: The factorization.

        Raises:
            ArithmeticError: If there is an unexpected problem preventing factoring, indicating a bug in the code.
        """
        if not self:
            return NonCommutativeFactorization(content=0,
                                               unit=hurwitzint(1, 0, 0, 0),
                                               primes=(),
                                               direction="right")

        # Extract integer content
        m = self.content()
        if m < 0:
            m = -m

        q = self
        if m > 1:
            q //= hurwitzint(m, 0, 0, 0)

        # Now q is primitive (or at least has no large integer content).
        n = abs(q)
        nf = _factorint(n)

        primes: list[hurwitzint] = []
        # Largest norm first, since the primes come off the right and get reversed below
        for p in sorted(nf.keys(), reverse=True):
            e = nf[p]
            for _ in range(e):
                pi = q._extract_right_prime(p)

                # divide on the right: q = qq * pi + 0
                qq, rr = divmod(q, pi)
                if rr:
                    raise ArithmeticError("extracted prime did not actually divide (unexpected)")
                q = qq
                primes.append(pi)

        # Remaining q must be a unit (norm 1) if we extracted all prime norms.
        if abs(q) != 1:
            raise ArithmeticError("remaining cofactor is not a unit; factorization incomplete")

        return NonCommutativeFactorization(content=m,
                                           unit=q,
                                           primes=tuple(reversed(primes)),
                                           direction="right")

    def factor_right(self) -> tuple[hurwitzint, ...]:
        """
        Return the Hurwitz primes of self as a plain tuple, whose product via `prod_right` is exactly `self`.

        This is `factor_right_detail().expand_content()` with the unit folded into the first prime, so every factor
            is a Hurwitz prime (its norm is a rational prime), including those of the content. That means factoring
            the content as an integer, which can be slow for a big content with big prime factors, and which
            `factor_right_detail` never does. Zero and the units have no prime factors, so they come back as themselves.

        Returns:
            tuple: The factors of self.
        """
        f = self.factor_right_detail().expand_content()
        if f.primes:
            return f.unit * f.primes[0], *f.primes[1:]

        return (self,)

    def factor_left_detail(self) -> NonCommutativeFactorization:
        """
        Deterministic left factorization normal form, the mirror image of `factor_right_detail`.

        Returns content, unit, and a tuple of Hurwitz primes P1..Pk, sorted by norm (smallest first), such that:
            self = content * Pk * ... * P2 * P1 * unit

        which is exactly what `prod_left` computes from the tuple. Primes are peeled off the left, largest norm first,
            and each is normalized via right-associates instead. (Same content logic.)

        Returns:
            NonCommutativeFactorization: The factorization.

        Raises:
            ArithmeticError: If there is a problem preventing factoring.
        """
        if not self:
            return NonCommutativeFactorization(content=0, unit=hurwitzint(1, 0, 0, 0), primes=(), direction="left")

        m = self.content()
        if m < 0:
            m = -m

        q = self
        if m > 1:
            q = q.rfloordiv(hurwitzint(m, 0, 0, 0))

        n = abs(q)
        nf = _factorint(n)

        primes: list[hurwitzint] = []
        # Largest norm first, since the primes come off the left and get reversed below
        for p in sorted(nf.keys(), reverse=True):
            e = nf[p]
            for _ in range(e):
                pi = q._extract_left_prime(p)

                # Divide on the left: q = pi * qq
                qq, rr = q.rdivmod(pi)
                if rr:
                    raise ArithmeticError("extracted left prime did not actually divide (unexpected)")
                q = qq
                primes.append(pi)

        if abs(q) != 1:
            raise ArithmeticError("remaining cofactor is not a unit; factorization incomplete")

        return NonCommutativeFactorization(content=m,
                                           unit=q,
                                           primes=tuple(reversed(primes)),
                                           direction="left")

    def factor_left(self) -> tuple[hurwitzint, ...]:
        """
        Return the Hurwitz primes of self as a plain tuple, whose product via `prod_left` is exactly `self`.

        This is `factor_left_detail().expand_content()` with the unit folded into the first prime, the mirror image
            of `factor_right` (see there for the cost of factoring the content).

        Note: `prod_left` multiplies factors on the *left* (so the iterable order is reversed
            in the final product). We return factors in the order that `prod_left` expects.

        Returns:
            tuple: The factors of self.
        """
        f = self.factor_left_detail().expand_content()
        if f.primes:
            return f.primes[0] * f.unit, *f.primes[1:]

        return (self,)
    # endregion


if not hurwitzint.UNITS:
    def units() -> list[hurwitzint]:
        """All the unit directions from the origin"""
        # ±1, ±i, ±j, ±k, and (±1±i±j±k)/2 (16 of them).
        out: list[hurwitzint] = []

        one = hurwitzint(1, 0, 0, 0)
        i = hurwitzint(0, 1, 0, 0)
        j = hurwitzint(0, 0, 1, 0)
        k = hurwitzint(0, 0, 0, 1)

        for s in (-1, 1):
            out.extend([s * one, s * i, s * j, s * k])

        # 16 half-units
        out.extend([
            hurwitzint(a, b, c, d, half=True)
            for a in (-1, 1)
            for b in (-1, 1)
            for c in (-1, 1)
            for d in (-1, 1)
        ])

        # Optional:
        out.sort(key=tuple)
        return out

    hurwitzint.UNITS = units()


def rdivmod(a: hurwitzint, b: OP_TYPES) -> tuple[hurwitzint, hurwitzint]:
    """Simply a helper method to match existing Python divmod syntax"""
    return a.rdivmod(b)


def gcd_left(a: hurwitzint, b: OP_TYPES) -> hurwitzint:
    """Simply a helper method to match existing Python gcd syntax"""
    return a.gcd_left(b)


def gcd_right(a: hurwitzint, b: OP_TYPES) -> hurwitzint:
    """Simply a helper method to match existing Python gcd syntax"""
    return a.gcd_right(b)


def xgcd_left(a: hurwitzint, b: OP_TYPES) -> tuple[hurwitzint, hurwitzint, hurwitzint]:
    """Simply a helper method to match gcd_left, for a.xgcd_left(b)"""
    return a.xgcd_left(b)


def xgcd_right(a: hurwitzint, b: OP_TYPES) -> tuple[hurwitzint, hurwitzint, hurwitzint]:
    """Simply a helper method to match gcd_right, for a.xgcd_right(b)"""
    return a.xgcd_right(b)


def prod_right(x: Iterable[OP_TYPES], start: OP_TYPES | None = None):
    """Simply a helper method to match existing Python prod syntax"""
    if start is None:
        start = 1

    return prod(x, start=start)


def prod_left(x: Iterable[OP_TYPES], start: OP_TYPES | None = None):
    """Simply a helper method to match existing Python prod syntax"""
    if start is None:
        start = 1

    for sub_x in x:
        start = sub_x * start

    return start
