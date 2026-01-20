from dataclasses import dataclass
from functools import cache
from math import gcd, prod
from typing import Callable, ClassVar, Iterable, Iterator, Literal, Optional, Union

from sympy import factorint

OTHER_OP_TYPES = Union[int, float]
_OTHER_OP_TYPES = (int, float)  # mypyc-friendly for isinstance
OP_TYPES = Union["hurwitzint", OTHER_OP_TYPES]


# TODO: Once Py3.9 support has been dropped, add slots=True
# @dataclass(frozen=True, slots=True)
@dataclass(frozen=True)
class HurwitzFactorization:
    """
    Canonical-ish normal form (deterministic):

        x = content * unit * P1 * P2 * ... * Pk

    - content is a positive integer (maximal integer dividing x).
    - unit is a canonical Hurwitz unit (norm 1).
    - Pi are Hurwitz primes (norm is a rational prime), each normalized by unit-migration.
    """
    content: int
    unit: "hurwitzint"
    primes: tuple["hurwitzint", ...]
    direction: Literal["left", "right"]

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


def _round_div_ties_away_from_zero(a: int, b: int) -> int:
    """Round a/b to nearest integer; ties go away from zero. b must be > 0."""
    if b <= 0:
        raise ValueError("b must be > 0")

    if a >= 0:
        return (a + (b // 2)) // b

    # a < 0
    return -((-a + (b // 2)) // b)


def mod_sqrt_prime(n: int, p: int) -> Optional[int]:
    """Return x such that x*x % p == n % p, or None if no sqrt exists. p must be prime."""
    n %= p
    if n == 0:
        return 0

    if p == 2:
        return n

    # Legendre symbol test: residue iff n^((p-1)/2) == 1 (mod p)
    if pow(n, (p - 1) // 2, p) != 1:
        return None

    # Fast path when p ≡ 3 (mod 4)
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    # Tonelli-Shanks
    q = p - 1
    s = 0
    while q % 2 == 0:
        s += 1
        q //= 2

    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while t != 1:
        i = 1
        t2i = (t * t) % p
        while i < m and t2i != 1:
            t2i = (t2i * t2i) % p
            i += 1

        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i

    return r


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
    """

    __slots__ = ("a", "b", "c", "d")

    a: int
    b: int
    c: int
    d: int

    UNITS: ClassVar[list["hurwitzint"]] = []

    def __init__(
        self,
        a: int = 0,
        b: int = 0,
        c: int = 0,
        d: int = 0,
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
            b: See a.
            c: See a.
            d: See a.
            half:
                Whether inputs are already in numerator-units for /2 representation.

        Raises:
            ValueError: If parity is incorrect.
        """
        a0, b0, c0, d0 = int(a), int(b), int(c), int(d)

        if not half:
            a0 *= 2
            b0 *= 2
            c0 *= 2
            d0 *= 2

        # All four must have the same parity.
        if ((a0 ^ b0) & 1) or ((a0 ^ c0) & 1) or ((a0 ^ d0) & 1):
            raise ValueError("For Hurwitz integers, a,b,c,d must all have the same parity")

        self.a, self.b, self.c, self.d = a0, b0, c0, d0

    # region constructors / conversions
    @classmethod
    def _make(cls, A: int, B: int, C: int, D: int) -> "hurwitzint":
        """Construct a new value of *this* conceptual type from internal numerators A,B,C,D."""
        return cls(A, B, C, D, half=True)

    @classmethod
    def _from_obj(cls, n: OP_TYPES) -> "hurwitzint":
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
        return ((self.a | self.b | self.c | self.d) & 1) == 0

    @property
    def den(self) -> int:
        """Static determinant, because everything is doubled under the hood anyway"""
        return 2

    def conjugate(self) -> "hurwitzint":
        """Quaternion conjugation: a+bi+cj+dk -> a-bi-cj-dk (in numerator units)."""
        return self._make(self.a, -self.b, -self.c, -self.d)

    def components2(self) -> tuple[int, int, int, int]:
        """Return the stored numerator components (A,B,C,D) for (...)/2."""
        return (self.a, self.b, self.c, self.d)

    def __add__(self, other: OP_TYPES) -> "hurwitzint":
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, hurwitzint):
            return self._make(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

        return NotImplemented

    def __radd__(self, other: OTHER_OP_TYPES) -> "hurwitzint":
        return self.__add__(other)

    def __sub__(self, other: OP_TYPES) -> "hurwitzint":
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if isinstance(other, hurwitzint):
            return self._make(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

        return NotImplemented

    def __rsub__(self, other: OTHER_OP_TYPES) -> "hurwitzint":
        return self.__neg__().__add__(other)

    def __neg__(self) -> "hurwitzint":
        return self._make(-self.a, -self.b, -self.c, -self.d)

    def __pos__(self) -> "hurwitzint":
        return self._make(self.a, self.b, self.c, self.d)

    def __mul__(self, other: OP_TYPES) -> "hurwitzint":
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            return NotImplemented

        # Quaternion multiplication in numerator units.
        # If q=(A+Bi+Cj+Dk)/2 and r=(E+Fi+Gj+Hk)/2,
        # then qr has denominator 4; we store with denominator 2,
        # so we must divide resulting numerators by 2.
        A, B, C, D = self.a, self.b, self.c, self.d
        E, F, G, H = other.a, other.b, other.c, other.d

        # (a,b,c,d)*(e,f,g,h) with i^2=j^2=k^2=ijk=-1:
        P = A * E - B * F - C * G - D * H
        Q = A * F + B * E + C * H - D * G
        R = A * G - B * H + C * E + D * F
        S = A * H + B * G - C * F + D * E

        # Must be divisible by 2 to land back in the Hurwitz order.
        if (P & 1) or (Q & 1) or (R & 1) or (S & 1):
            raise ArithmeticError("Non-integral product; parity constraint violated")

        return self._make(P // 2, Q // 2, R // 2, S // 2)

    def __rmul__(self, other: OTHER_OP_TYPES) -> "hurwitzint":
        return self.__mul__(other)

    def __pow__(self, exp: int) -> "hurwitzint":
        e = int(exp)
        if e < 0:
            raise ValueError("Negative powers not supported")

        result = hurwitzint(1, 0, 0, 0)  # multiplicative identity
        base: hurwitzint = self
        while e:
            if e & 1:
                result = result * base

            e >>= 1
            if e:
                base = base * base

        return result

    # region Euclidean division (Hurwitz order is norm-Euclidean)
    def _division(
            self,
            num: "hurwitzint",
            divisor: "hurwitzint",
            divisor_norm: int,
            remainder: Callable[["hurwitzint", "hurwitzint", "hurwitzint"], "hurwitzint"] = lambda a, b, c: a - b * c) \
            -> tuple["hurwitzint", "hurwitzint"]:
        """
        A shared division algorithm.

        Args:
            num: The number to divide.
            divisor: The divisor.
            divisor_norm: The divisor norm. Should be checked for 0 already!
            remainder: A callable to compute the remainder.

        Returns:
            tuple: The quotient and remainder.
        """

        # We want q ≈ num / n, but everything is stored in numerator-units (../2).
        # If num is stored as (U)/2, and we want q stored as (Q)/2, then Q ≈ U / n.
        A0 = _round_div_ties_away_from_zero(num.a, divisor_norm)
        B0 = _round_div_ties_away_from_zero(num.b, divisor_norm)
        C0 = _round_div_ties_away_from_zero(num.c, divisor_norm)
        D0 = _round_div_ties_away_from_zero(num.d, divisor_norm)

        # Choose best among a small neighborhood under Euclidean metric in R^4,
        # but restricted to Hurwitz parity lattice: all components same parity.
        bestA, bestB, bestC, bestD = A0, B0, C0, D0
        best_metric: Optional[int] = None

        # metric is scaled distance:
        # sum_i (Qi*n - Ui)^2   (scale by n^2 doesn't matter for argmin)
        for A in (A0 - 1, A0, A0 + 1):
            for B in (B0 - 1, B0, B0 + 1):
                if ((A ^ B) & 1) != 0:
                    continue
                dA = A * divisor_norm - num.a
                dB = B * divisor_norm - num.b
                dA2 = dA * dA
                dB2 = dB * dB

                for C in (C0 - 1, C0, C0 + 1):
                    if ((A ^ C) & 1) != 0:
                        continue
                    dC = C * divisor_norm - num.c
                    dC2 = dC * dC

                    for D in (D0 - 1, D0, D0 + 1):
                        if ((A ^ D) & 1) != 0:
                            continue

                        dD = D * divisor_norm - num.d
                        metric = dA2 + dB2 + dC2 + (dD * dD)

                        if best_metric is None or metric < best_metric:
                            best_metric = metric
                            bestA, bestB, bestC, bestD = A, B, C, D

        q = self._make(bestA, bestB, bestC, bestD)
        r = remainder(self, q, divisor)
        return q, r

    # region Left-division helpers (non-commutative!)
    def __divmod__(self, other: OP_TYPES) -> tuple["hurwitzint", "hurwitzint"]:
        """
        Nearest-lattice division in the Hurwitz quaternion order.

        We define quotient on the LEFT (Python-style):
            self = q * other + r

        Because multiplication is non-commutative, this is a specific choice.

        Returns:
            (q, r) where r has small norm (typically < abs(other)).

        Raises:
            ZeroDivisionError: if other == 0
            NotImplementedError: if other is unsupported type
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise NotImplementedError

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        # num approximates self / other in the usual quaternion sense:
        # q ~ self * conj(other) / N(other)
        num = self * other.conjugate()

        return self._division(num, other, n)

    def __truediv__(self, other: OP_TYPES) -> "hurwitzint":
        # mirror QuadInt: treat / as Euclidean division in this domain
        return self.__floordiv__(other)

    def __rtruediv__(self, other: OTHER_OP_TYPES) -> "hurwitzint":
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__truediv__(self)

        return NotImplemented

    def __floordiv__(self, other: OP_TYPES) -> "hurwitzint":
        q, _ = divmod(self, other)
        return q

    def __rfloordiv__(self, other: OTHER_OP_TYPES) -> "hurwitzint":
        if isinstance(other, _OTHER_OP_TYPES):
            new_other = self._from_obj(other)
            return new_other.__floordiv__(self)

        return NotImplemented

    def __mod__(self, other: OP_TYPES) -> "hurwitzint":
        _, r = divmod(self, other)
        return r
    # endregion

    # region Right-division helpers
    def rdivmod(self, other: OP_TYPES) -> tuple["hurwitzint", "hurwitzint"]:
        """
        Right-quotient division in the Hurwitz quaternion order.

        Defines quotient on the RIGHT:
            self = other * q + r

        Raises:
            ZeroDivisionError: If trying to divide by 0.

        Returns:
            (q, r)
        """
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise NotImplementedError

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        # Right quotient: q ~ other^{-1} * self = conj(other) * self / N(other)
        num = other.conjugate() * self

        return self._division(num, other, n, lambda a, b, c: a - c * b)

    def rtruediv(self, other: OP_TYPES) -> "hurwitzint":
        """A version of __truediv__ for right-division"""
        return self.rfloordiv(other)

    def rfloordiv(self, other: OP_TYPES) -> "hurwitzint":
        """A version of __floordiv__ for right-division"""
        q, _ = self.rdivmod(other)
        return q

    def rmod(self, other: OP_TYPES) -> "hurwitzint":
        """A version of __mod__ for right-division"""
        _, r = self.rdivmod(other)
        return r
    # endregion
    # endregion

    def __abs__(self) -> int:
        """
        Reduced norm:
            N((A+Bi+Cj+Dk)/2) = (A^2+B^2+C^2+D^2)/4

        Always an integer for valid Hurwitz integers.

        Raises:
            ArithmeticError: If there is a non-integral norm due to parity violation.

        Returns:
            int: The norm.
        """
        num = self.a * self.a + self.b * self.b + self.c * self.c + self.d * self.d
        q, r = divmod(num, 4)
        if r != 0:
            raise ArithmeticError("Non-integral norm; parity constraint violated")

        return q

    def __bool__(self) -> bool:
        return (self.a | self.b | self.c | self.d) != 0

    def __iter__(self) -> Iterator[int]:
        return iter((self.a, self.b, self.c, self.d))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> int:
        if idx == 0:
            return self.a
        if idx == 1:
            return self.b
        if idx == 2:
            return self.c
        if idx == 3:
            return self.d
        raise IndexError("hurwitzint index out of range (valid: 0..3)")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, hurwitzint):
            return False

        return (self.a, self.b, self.c, self.d) == (other.a, other.b, other.c, other.d)

    def __hash__(self) -> int:
        return hash((self.a, self.b, self.c, self.d))

    def __repr__(self) -> str:
        if self.is_lipschitz:
            # If all even, show integer components without "/2".
            ra, rb, rc, rd = self.a // 2, self.b // 2, self.c // 2, self.d // 2
            den = None
        else:
            # Otherwise show numerator form "(...)/2".
            ra, rb, rc, rd = self.a, self.b, self.c, self.d
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

    # region GCD
    def _normalize_unit(self) -> "hurwitzint":
        """
        Deterministic associate choice up to ±1.

        Full normalization up to the 24 Hurwitz units is possible, but this keeps things
        cheap and stable: multiply by -1 so the first nonzero numerator component is > 0.

        Returns:
            hurwitzint: The unit normalized hurwitzint.
        """
        if not self:
            return self

        a, b, c, d = self
        if a != 0:
            return -self if a < 0 else self
        if b != 0:
            return -self if b < 0 else self
        if c != 0:
            return -self if c < 0 else self
        return -self if d < 0 else self

    def _gcd(self,
             other: OP_TYPES,
             *,
             divmod_method: Callable = divmod,
             normalize: bool = True) -> "hurwitzint":
        """GCD via Euclidean algorithm."""
        if isinstance(other, _OTHER_OP_TYPES):
            other = self._from_obj(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"Unable to divide hurwitzint and type {type(other)}")

        a = self
        b = other

        if not a:
            return b._normalize_unit() if normalize else b

        if b:
            last = abs(b)
            while b:
                _, r = divmod_method(a, b)
                a, b = b, r

                if b:
                    nb = abs(b)
                    if nb >= last:
                        raise ArithmeticError("Euclidean descent failed (non-decreasing remainder norm)")
                    last = nb

        return a._normalize_unit() if normalize else a

    def gcd_right(self,
                  other: OP_TYPES,
                  *,
                  normalize: bool = True) -> "hurwitzint":
        """
        Right gcd via left-division Euclidean algorithm.

        The result g is a "right gcd":
            a = q*b + r
            a = a' * g and b = b' * g   (up to multiplication by a unit)

        Returns:
            hurwitzint: The gcd.
        """
        return self._gcd(other, normalize=normalize)

    def gcd_left(self,
                 other: OP_TYPES,
                 *,
                 normalize: bool = True) -> "hurwitzint":
        """
        Left gcd via RIGHT-division Euclidean algorithm.

        The result g is a "left gcd":
            a = b*q + r
            a = g * a' and b = g * b'   (up to multiplication by a unit)

        Returns:
            hurwitzint: The gcd.
        """
        return self._gcd(other, divmod_method=rdivmod, normalize=normalize)
    # endregion

    # region Factoring
    def content(self) -> int:
        """
        Largest positive integer m such that self = m*q' with q' still a Hurwitz integer.
            Computed in numerator-units with the Hurwitz parity constraint.

        Returns:
            int: Computed content value.
        """
        A, B, C, D = self.a, self.b, self.c, self.d
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

    @staticmethod
    @cache
    def _find_uv_for_prime(p: int) -> tuple[int, int]:
        """
        Find u,v with 1 + u^2 + v^2 ≡ 0 (mod p).
            Deterministic search over u with Tonelli sqrt for v.

        Returns:
            tuple: The found u,v where 1 + u^2 + v^2 ≡ 0 (mod p).

        Raises:
            ArithmeticError: If we fail to find a good u,v pair.
        """
        p = int(p)
        if p == 2:
            return 0, 1

        for u in range(p):
            t = (-1 - (u * u)) % p
            v = mod_sqrt_prime(t, p)
            if v is not None:
                return u, v

        raise ArithmeticError("Failed to find u,v (unexpected for prime p)")

    @staticmethod
    @cache
    def _prime_over_rational(p: int) -> "hurwitzint":
        """
        Build a deterministic Hurwitz prime with norm p by Euclid from (p, 1+ui+vj).
            This mirrors the standard construction used in proofs of 4-squares and irreducibles.

        Returns:
            hurwitzint: A Hurwitz prime with norm p.

        Raises:
            ArithmeticError: If we fail to find a Hurwitz prime with norm p.
        """
        p = int(p)

        if p == 2:
            # A simple norm-2 prime
            base = hurwitzint(1, 1, 0, 0)  # norm 2
        else:
            u, v = hurwitzint._find_uv_for_prime(p)
            q = hurwitzint(1, u, v, 0)  # Lipschitz element in H(Z)
            base = hurwitzint.gcd_right(hurwitzint(p, 0, 0, 0), q)

        if abs(base) != p:
            raise ArithmeticError("prime construction failed: gcd did not have norm p")

        # Canonicalize base up to left associates (unit migration friendly for right factoring)
        base_canon, _ = base._canonical_associate(direction="left")
        return base_canon

    @staticmethod
    def _canonicalize_norm(factors: "HurwitzFactorization") -> "HurwitzFactorization":
        """
        Canonicalize the prime list by sorting primes by rational norm using metacommutation swaps.

        This is exactly the step needed if we want a stable canonical ordering of prime factors
            in a non-commutative setting.

        Returns:
            HurwitzFactorization: The canonicalized factorization.
        """
        primes = list(factors.primes)
        i = 0
        while i < len(primes) - 1:
            if abs(primes[i]) > abs(primes[i + 1]):
                primes[i], primes[i + 1] = primes[i]._metacommutate_pair(primes[i + 1], factors.direction)
                if i:
                    i -= 1  # bubble backward
            else:
                i += 1

        return HurwitzFactorization(content=factors.content,
                                    unit=factors.unit,
                                    primes=tuple(primes),
                                    direction=factors.direction)

    def _metacommutate_pair(self,
                            q: "hurwitzint",
                            direction: Literal["left", "right"]) -> tuple["hurwitzint", "hurwitzint"]:
        """
        Metacommutation step for adjacent primes of *distinct* rational norms.

        Given p, q (typically Hurwitz primes) with abs(p) != abs(q), find p2, q2 such that:
            p * q == q2 * p2

        Returns:
            (q2, p2)  # i.e. swapped pair, adjusted by associates so the product is preserved.

        Raises:
            ArithmeticError: if no metacommutation swap is found
                (should not happen for genuine primes of distinct norms).
            ValueError: Failed to metacommutate pair due to poor input?
        """
        if abs(self) == abs(q):
            raise ValueError("metacommutation requires distinct rational norms")

        if direction == "right":
            pq = self * q
            divmod_method = rdivmod
        else:
            pq = q * self
            divmod_method = divmod

        best_key = None
        best_q = None
        best_p = None

        # Search associates of q: uL * q * uR, try to left-divide pq by that candidate
        for uL in hurwitzint.UNITS:
            for uR in hurwitzint.UNITS:
                cand_q = uL * q * uR
                cand_p, r = divmod_method(pq, cand_q)   # pq = cand_q * cand_p + r

                if r:
                    continue
                if abs(cand_p) != abs(self):
                    continue

                key = (cand_q.components2(), cand_p.components2())
                if best_key is None or key < best_key:
                    best_key = key
                    best_q = cand_q
                    best_p = cand_p

        if best_q is None or best_p is None:
            raise ArithmeticError("metacommutation failed to find a swap")

        # Canonicalize the LEFT factor by right-associate normalization, and migrate the unit across the boundary:
        #   (best_q * u) * (u^{-1} * best_p) == best_q * best_p
        q_canon, u = best_q._canonical_associate(direction)

        # u^{-1} == conjugate(u) for units
        p_adj = u.conjugate() * best_p if direction == "right" else best_p * u.conjugate()

        return q_canon, p_adj

    def _canonical_associate(self, direction: Literal["left", "right"]) -> tuple["hurwitzint", "hurwitzint"]:
        """
        Unit-migration normalization for a *left* factor:
            replace p by p*u (u a unit).

        Returns:
             tuple: (p_canon, u) such that p_canon = p*u.
        """
        best = None
        best_u = None
        for u in hurwitzint.UNITS:
            cand = self * u if direction == "right" else u * self
            key = cand.components2()
            if best is None or key < best:
                best = key
                best_u = u

        assert best_u is not None
        if direction == "right":
            return self * best_u, best_u
        return best_u * self, best_u

    def _extract_right_prime(self, p: int) -> "hurwitzint":
        """
        Extract a right prime of norm p dividing self.
            g = gcd_right(self, p)

        Returns:
            hurwitzint: A Hurwitz prime with norm p that divides self.

        Raises:
            ArithmeticError: If there is an unexpected problem preventing factoring, indicating a bug in the code.
        """
        # Directly extract from q via gcd with the scalar p
        scalar = hurwitzint(p, 0, 0, 0)
        g = self.gcd_right(scalar, normalize=False)

        if abs(g) != p:
            raise ArithmeticError(f"Failed to extract right prime for {p=}")

        g_canon, _ = g._canonical_associate(direction="left")
        return g_canon

    def factor_right(self) -> HurwitzFactorization:
        """
        Deterministic right factorization normal form (primitive-first).

        Returns content, unit, and a tuple of Hurwitz primes P1..Pk such that:
            self = P1 * P2 * ... * Pk * content * unit

        Notes:
          - We do NOT expand `content` into Hurwitz primes by default because scalar
            factorization is exactly where recombination/nonuniqueness explodes.
          - Each Pi is irreducible because its norm is a rational prime.

        Returns:
            HurwitzFactorization: The factorization.

        Raises:
            ArithmeticError: If there is an unexpected problem preventing factoring, indicating a bug in the code.
        """
        if not self:
            return HurwitzFactorization(content=0,
                                        unit=hurwitzint(1, 0, 0, 0),
                                        primes=(),
                                        direction="right")

        # Extract integer content
        m = self.content()
        if m < 0:
            m = -m

        q = self
        if m > 1:
            q, r = divmod(q, hurwitzint(m, 0, 0, 0))
            if r:
                # Shouldn't happen if content() is correct
                raise ArithmeticError("content division produced remainder")

        # Now q is primitive (or at least has no large integer content).
        n = abs(q)
        nf = factorint(n)

        primes: list[hurwitzint] = []
        for p in sorted(nf.keys()):
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

        return self._canonicalize_norm(HurwitzFactorization(content=m,
                                       unit=q,
                                       primes=tuple(reversed(primes)),
                                       direction="right"))

    def factor_left(self) -> HurwitzFactorization:
        """
        Deterministic left factorization normal form.

        Produces:
            self = content * (P1 * P2 * ... * Pk) * unit

        (Same content logic; primes are normalized via right-associates instead.)

        Returns:
            HurwitzFactorization: The factorization.

        Raises:
            ArithmeticError: If there is a problem preventing factoring.
        """
        if not self:
            return HurwitzFactorization(content=0, unit=hurwitzint(1, 0, 0, 0), primes=(), direction="left")

        m = self.content()
        if m < 0:
            m = -m

        q = self
        if m > 1:
            q, r = q.rdivmod(hurwitzint(m, 0, 0, 0))
            if r:
                raise ArithmeticError("content division produced remainder")

        n = abs(q)
        nf = factorint(n)

        primes: list[hurwitzint] = []
        for p in sorted(nf.keys()):
            e = nf[p]
            for _ in range(e):
                # Extract a LEFT prime factor of norm p:
                # do the symmetric trick by factoring the conjugate on the right,
                # then conjugating back.
                pi_r = q.conjugate()._extract_right_prime(p).conjugate()

                # Normalize for left-factorization by right-multiplying a unit: pi = pi_r * u
                pi, _ = pi_r._canonical_associate(direction="right")

                # Divide on the left: q = pi * qq
                qq, rr = q.rdivmod(pi)
                if rr:
                    raise ArithmeticError("extracted left prime did not actually divide (unexpected)")
                q = qq
                primes.append(pi)

        if abs(q) != 1:
            raise ArithmeticError("remaining cofactor is not a unit; factorization incomplete")

        return self._canonicalize_norm(HurwitzFactorization(content=m,
                                       unit=q,
                                       primes=tuple(reversed(primes)),
                                       direction="left"))
    # endregion


if not hurwitzint.UNITS:
    def units() -> list["hurwitzint"]:
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
        out.sort(key=lambda u: u.components2())
        return out

    hurwitzint.UNITS = units()


def rdivmod(a: "hurwitzint", b: OP_TYPES) -> tuple["hurwitzint", "hurwitzint"]:
    """Simply a helper method to match existing Python divmod syntax"""
    return a.rdivmod(b)


def gcd_left(a: "hurwitzint", b: OP_TYPES) -> "hurwitzint":
    """Simply a helper method to match existing Python gcd syntax"""
    return a.gcd_left(b)


def gcd_right(a: "hurwitzint", b: OP_TYPES) -> "hurwitzint":
    """Simply a helper method to match existing Python gcd syntax"""
    return a.gcd_right(b)


def prod_right(x: Iterable[OP_TYPES], start: Union[OP_TYPES, None] = None):
    """Simply a helper method to match existing Python prod syntax"""
    if start is None:
        start = 1

    return prod(x, start=start)


def prod_left(x: Iterable[OP_TYPES], start: Union[OP_TYPES, None] = None):
    """Simply a helper method to match existing Python prod syntax"""
    if start is None:
        start = 1

    for sub_x in x:
        start = sub_x * start

    return start
