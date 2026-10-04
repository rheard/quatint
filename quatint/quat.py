from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import gcd, prod
from typing import ClassVar, Iterable, Iterator, Literal, TypeGuard

from sympy import factorint, isprime

def _is_number(x: object) -> TypeGuard[int | float]:
    """
    Return whether x is an int or a float (so a bool too), the plain numbers that hurwitzint takes everywhere.

    This is isinstance(x, (int, float)) in two halves, since mypyc compiles isinstance against one builtin type to a
        quick type check, but against a tuple to a generic isinstance call, which is far slower. Every operation that
        can take a number makes this check.

    Returns:
        bool: Whether x is an int or float.
    """
    return isinstance(x, int) or isinstance(x, float)  # ruff: ignore[duplicate-isinstance-call]


def _part_to_int(x: int | float) -> int:
    """
    Return int(x), for a part given to hurwitzint() that isn't a plain int: a float, truncated, or a bool.

    Returns:
        int: The part, as an int.

    Raises:
        TypeError: If x is not an int or float.
    """
    # The mypyc build rejects anything else before getting here, so this makes pure Python match it, rather than
    #   int() a string, a Fraction or a real hurwitzint
    if not _is_number(x):
        raise TypeError(f"hurwitzint parts must be int or float, not {type(x).__name__!r}")

    return int(x)


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
            TypeError: If factors is not a dict, or has anything but ints in it.
            ValueError: If factors has a negative exponent, does not multiply to content, or has a key that is not
                prime.
        """
        m = self.content
        if factors is not None:
            # The mypyc build rejects anything but a dict of ints before getting here, so these make pure Python match
            #   it (a dict subclass like Counter is fine in both)
            if not isinstance(factors, dict):
                raise TypeError(f"factors must be a dict, not {type(factors).__name__!r}")

            if not all(isinstance(p, int) and isinstance(e, int) for p, e in factors.items()):
                raise TypeError("factors must map int primes to int exponents")

            # Before the product, since a negative exponent would make it a float, and that can round to exactly the
            #   content: 2**60 * 3**-1 rounds to 384307168202282304, which would then expand as if it were 2**60
            if any(e < 0 for e in factors.values()):
                raise ValueError("factors can't have negative exponents")

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

    def prod(self) -> hurwitzint:
        """Recreate the number using prod_right or prod_left"""
        if self.direction == "right":
            return self.prod_right()
        return self.prod_left()

    def prod_right(self) -> hurwitzint:
        """Recreate the number using prod_right"""
        return prod_right(self.primes, start=self.unit * self.content)

    def prod_left(self) -> hurwitzint:
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


def _nearest_by_parity(U: int, n: int) -> tuple[int, int, int, int]:
    """
    Return the even integer nearest U/n with its distance from U/n, then the odd one with its distance (n > 0).

    The distances are in units of 1/n, so they are the integers |U - n*Q|. With fl = U // n, U/n lies in [fl, fl + 1),
        so the nearest integer of fl's parity is fl itself, and the nearest of the other parity is fl + 1, except when
        U/n == fl exactly: then fl - 1 is just as near, both n away. That is the only way two can tie, so a distance of
        n always means a tie, and then this returns the smaller one, fl - 1.

    Returns:
        tuple: (even integer, its distance, odd integer, its distance).
    """
    fl = U // n
    rem = U - _mul(fl, n)
    if fl & 1:
        if rem:
            return fl + 1, n - rem, fl, rem

        return fl - 1, n, fl, 0

    if rem:
        return fl, rem, fl + 1, n - rem

    return fl, 0, fl - 1, n


def _mul_numerators(A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> tuple[int, int, int, int]:
    """
    Multiply two Hurwitz integers given as numerators, (A+Bi+Cj+Dk)/2 * (E+Fi+Gj+Hk)/2.

    hurwitzint.__mul__ is just this. The division code calls it directly, so it builds no hurwitzint (or parity check)
        for an intermediate product. The product of two Hurwitz integers is one too, so each numerator over 4 is even
        and halves exactly.

    Returns:
        tuple: The product's numerators, over 2.
    """
    return ((_mul(A, E) - _mul(B, F) - _mul(C, G) - _mul(D, H)) // 2,
            (_mul(A, F) + _mul(B, E) + _mul(C, H) - _mul(D, G)) // 2,
            (_mul(A, G) - _mul(B, H) + _mul(C, E) + _mul(D, F)) // 2,
            (_mul(A, H) + _mul(B, G) - _mul(C, F) + _mul(D, E)) // 2)


def _divmod_numerators(A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int, n: int,
                       *,
                       right: bool,
                       canonical: bool = True) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Nearest-lattice division of x = (A+Bi+Cj+Dk)/2 by y = (E+Fi+Gj+Hk)/2 (with n = N(y) > 0), all as numerators.

    Chooses the Hurwitz integer q that leaves the remainder r = x - q*y (or x - y*q if right) of least norm. With U the
        numerators of x * conj(y) (or conj(y) * x if right), r * conj(y) == x * conj(y) - n*q, so that means minimizing
            sum_i (Ui - n*Qi)^2
        over numerators Qi that share one parity. Each parity's parts are independent, so _nearest_by_parity picks
        each one, and the parity with the smaller sum wins.

    When several q leave a remainder of that least norm, this takes the one whose remainder has the largest numerator
        tuple (see _tied_division). That only depends on the remainders, which are the same for every x
        congruent modulo the multiples of y (x + h*y has the same ones, from q + h), so x % y is the same for all of
        them too. For an integer y, that makes x % y the canonical residue that pow(x, e, y) and inv_mod give.
        With canonical=False it takes any of them, for the gcds: Euclid's algorithm only needs a remainder that small,
        and ties are common in the small divisions near its end, which factorization runs a lot of.

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

    Ea, dEa, Oa, dOa = _nearest_by_parity(Ua, n)
    Eb, dEb, Ob, dOb = _nearest_by_parity(Ub, n)
    Ec, dEc, Oc, dOc = _nearest_by_parity(Uc, n)
    Ed, dEd, Od, dOd = _nearest_by_parity(Ud, n)

    # The even quotient leaves 4n*N(r) == sum(d^2) over its distances d, and the odd one sum((n - d)^2), since each
    #   part's two distances add up to n. Their difference is 2n * (2n - sum(d)), so the even one leaves the smaller
    #   remainder exactly when its distances add up to less than 2n, and the two tie at 2n
    two_n = n + n
    even_sum = dEa + dEb + dEc + dEd

    # A distance of n is a part with two nearest numerators (see _nearest_by_parity), so when the parity that wins
    #   outright has none, its quotient is the only one of least norm
    if even_sum < two_n and dEa != n and dEb != n and dEc != n and dEd != n:
        Qa, Qb, Qc, Qd = Ea, Eb, Ec, Ed
    elif even_sum > two_n and dOa != n and dOb != n and dOc != n and dOd != n:
        Qa, Qb, Qc, Qd = Oa, Ob, Oc, Od
    elif not canonical:
        # A tie, where any quotient of least norm will do
        Qa, Qb, Qc, Qd = (Ea, Eb, Ec, Ed) if even_sum <= two_n else (Oa, Ob, Oc, Od)
    else:
        # A tie, where picking the remainder takes more work, which _tied_division does (giving the remainder too)
        return _tied_division((A, B, C, D), (E, F, G, H), (Ea, Eb, Ec, Ed), (dEa, dEb, dEc, dEd),
                              (Oa, Ob, Oc, Od), (dOa, dOb, dOc, dOd), n,
                              use_even=even_sum <= two_n, use_odd=even_sum >= two_n, right=right)

    # The remainder is x - q*y (or x - y*q)
    if right:
        Pa, Pb, Pc, Pd = _mul_numerators(E, F, G, H, Qa, Qb, Qc, Qd)
    else:
        Pa, Pb, Pc, Pd = _mul_numerators(Qa, Qb, Qc, Qd, E, F, G, H)

    return Qa, Qb, Qc, Qd, A - Pa, B - Pb, C - Pc, D - Pd


def _tied_division(x: tuple[int, int, int, int],
                   y: tuple[int, int, int, int],
                   even: tuple[int, int, int, int],
                   even_distances: tuple[int, int, int, int],
                   odd: tuple[int, int, int, int],
                   odd_distances: tuple[int, int, int, int],
                   n: int,
                   *,
                   use_even: bool,
                   use_odd: bool,
                   right: bool) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Finish _divmod_numerators when several quotients leave a remainder of the least norm, by taking the one whose
        remainder has the largest numerator tuple, as _canonical_associate takes the largest associate.

    It has to be a rule about the remainder for x % y to be the same for every x congruent modulo the multiples of y:
        x + h*y shifts every quotient by h, and leaves the remainders as they were.

    The candidates are the quotient of each parity that leaves the least norm (use_even and use_odd), from
        _nearest_by_parity, where each part that ties (its distance is n) takes either nearest numerator: the one given,
        or the one 2 above.

    Returns:
        tuple: The numerators of the quotient, then those of its remainder.
    """
    found = False
    Qa = Qb = Qc = Qd = Ra = Rb = Rc = Rd = 0
    for parity in range(2):  # Not over (0, 1), which mypyc iterates as a generic Python object
        if not (use_odd if parity else use_even):
            continue

        qa, qb, qc, qd = odd if parity else even
        da, db, dc, dd = odd_distances if parity else even_distances

        # The parts that tie, as bits, and then every subset of them (each mask, from all of them down to none, moves
        #   its parts to their other nearest numerator, 2 above)
        tied = (1 if da == n else 0) | (2 if db == n else 0) | (4 if dc == n else 0) | (8 if dd == n else 0)
        mask = tied
        while True:
            q = (qa + 2 * (mask & 1), qb + (mask & 2), qc + ((mask & 4) >> 1), qd + ((mask & 8) >> 2))
            ra, rb, rc, rd = _sub_mul_numerators(x, q, y, right=right)

            # One part at a time, since mypyc compiles a tuple comparison into a slow generic one
            if not found:
                larger = True
            elif ra != Ra:
                larger = ra > Ra
            elif rb != Rb:
                larger = rb > Rb
            elif rc != Rc:
                larger = rc > Rc
            else:
                larger = rd > Rd

            if larger:
                found = True
                Qa, Qb, Qc, Qd = q
                Ra, Rb, Rc, Rd = ra, rb, rc, rd

            if not mask:
                break

            mask = (mask - 1) & tied

    return Qa, Qb, Qc, Qd, Ra, Rb, Rc, Rd


def _sub_mul_numerators(x: tuple[int, int, int, int],
                        q: tuple[int, int, int, int],
                        y: tuple[int, int, int, int],
                        *,
                        right: bool) -> tuple[int, int, int, int]:
    """
    Return x - q*y (or x - y*q if right), with each Hurwitz integer given as its numerators, like _mul_numerators.

    This works out the remainders of tied quotients (see _tied_division), and steps the Bezout
        coefficients of the extended gcd along with its remainders (see hurwitzint._xgcd).

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


def _residue_numerators(A: int, B: int, C: int, D: int, m: int) -> tuple[int, int, int, int]:
    """
    Return x % m for x = (A+Bi+Cj+Dk)/2 and an integer m > 0, as numerators, for inv_mod and pow(x, e, m).

    That is the remainder _divmod_numerators leaves (the one of least norm, with ties going to the largest numerator
        tuple), worked out with much less arithmetic, since the divisor is an integer. Then the remainder's parts are
        just A - m*Qa and so on, so in each part, the tie between two nearest numerators goes to the smaller one, which
        _nearest_by_parity gives, and a tie between the parities to the smaller first numerator. (The first parts
        always differ, by m times an odd number.) Which parity leaves the smaller remainder comes from the sum of the
        even distances, as in _divmod_numerators.

    Returns:
        tuple: The remainder's numerators, over 2.
    """
    Ea, dEa, Oa, _ = _nearest_by_parity(A, m)
    Eb, dEb, Ob, _ = _nearest_by_parity(B, m)
    Ec, dEc, Oc, _ = _nearest_by_parity(C, m)
    Ed, dEd, Od, _ = _nearest_by_parity(D, m)

    two_m = m + m
    even_sum = dEa + dEb + dEc + dEd
    if even_sum < two_m or (even_sum == two_m and Ea < Oa):
        return A - _mul(m, Ea), B - _mul(m, Eb), C - _mul(m, Ec), D - _mul(m, Ed)

    return A - _mul(m, Oa), B - _mul(m, Ob), C - _mul(m, Oc), D - _mul(m, Od)


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
        a: int | float = 0,
        b: int | float = 0,
        c: int | float = 0,
        d: int | float = 0,
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
                Floats are truncated with int(), like everywhere else a float meets a hurwitzint, and anything but
                an int or float raises TypeError.
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
        a0 = a if type(a) is int else _part_to_int(a)
        b0 = b if type(b) is int else _part_to_int(b)
        c0 = c if type(c) is int else _part_to_int(c)
        d0 = d if type(d) is int else _part_to_int(d)

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
    @staticmethod
    def _make(A: int, B: int, C: int, D: int) -> hurwitzint:
        """Construct a hurwitzint from its numerators, as every arithmetic result (and every unpickled one) is."""
        # By the class's own name, mypyc calls the native constructor directly. Through cls it was a generic Python
        #   call, which parsed the arguments (matching half up by name) before getting there, and that was about half
        #   of what a + b cost. A compiled hurwitzint can't be subclassed anyway.
        return hurwitzint(A, B, C, D, half=True)
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

    def __add__(self, other: hurwitzint | int | float) -> hurwitzint:
        if isinstance(other, hurwitzint):
            return self._make(self._a + other._a, self._b + other._b, self._c + other._c, self._d + other._d)

        if _is_number(other):
            # A number only moves the real part, whose numerator is twice the number: n + n, since mypyc's int multiply
            #   leaves its fast path for a negative n (see _mul)
            n = other if type(other) is int else int(other)
            return self._make(self._a + n + n, self._b, self._c, self._d)

        return NotImplemented

    def __radd__(self, other: int | float) -> hurwitzint:
        return self.__add__(other)

    def __sub__(self, other: hurwitzint | int | float) -> hurwitzint:
        if isinstance(other, hurwitzint):
            return self._make(self._a - other._a, self._b - other._b, self._c - other._c, self._d - other._d)

        if _is_number(other):
            n = other if type(other) is int else int(other)
            return self._make(self._a - n - n, self._b, self._c, self._d)

        return NotImplemented

    def __rsub__(self, other: int | float) -> hurwitzint:
        # other - self, without making -self first
        if _is_number(other):
            n = other if type(other) is int else int(other)
            return self._make(n + n - self._a, -self._b, -self._c, -self._d)

        return NotImplemented

    def __neg__(self) -> hurwitzint:
        return self._make(-self._a, -self._b, -self._c, -self._d)

    def __pos__(self) -> hurwitzint:
        return self._make(self._a, self._b, self._c, self._d)

    def __mul__(self, other: hurwitzint | int | float) -> hurwitzint:
        if isinstance(other, hurwitzint):
            P, Q, R, S = _mul_numerators(self._a, self._b, self._c, self._d, other._a, other._b, other._c, other._d)
            return self._make(P, Q, R, S)

        if _is_number(other):
            # A number scales every part: 4 products, rather than a quaternion product's 16
            n = other if type(other) is int else int(other)
            return self._make(_mul(self._a, n), _mul(self._b, n), _mul(self._c, n), _mul(self._d, n))

        return NotImplemented

    def __rmul__(self, other: int | float) -> hurwitzint:
        return self.__mul__(other)

    # Not just `exp: float`, since mypyc would turn an int exponent into a double and lose every bit past 2**53
    def __pow__(self, exp: int | float, mod: hurwitzint | int | float | None = None) -> hurwitzint:
        """
        Return self**exp, or for pow(self, exp, mod), self**exp modulo the integer mod.

        Without a modulus, a negative exp only works for a unit, whose inverse is its conjugate.

        With one, the result is what % leaves, like that of inv_mod: the canonical residue of its class, the one of
            least norm with ties going to the largest numerator tuple. So pow(x, e, mod) == (x**e) % mod, as for an int,
            and congruent values have the very same powers. A negative exp takes powers of self.inv_mod(mod), as
            Python's pow does for an int (so it raises ValueError when there is no inverse). The modulus has to be an
            integer, as for inv_mod: a float is truncated with int(), a hurwitzint has to be real, and 0 raises
            ZeroDivisionError.

        Returns:
            hurwitzint: The power.

        Raises:
            ValueError: If exp is negative without a modulus, and self is not a unit.
        """
        # The mypyc build rejects anything else before getting here, so this makes pure Python match it
        if not _is_number(exp):
            return NotImplemented

        e = int(exp)
        if mod is not None:
            if not (isinstance(mod, hurwitzint) or _is_number(mod)):
                return NotImplemented

            return self._pow_mod(e, self._modulus(mod, "pow()"))

        base: hurwitzint = self
        if e < 0:
            # x**-n is (x**-1)**n, which is only a Hurwitz integer when x is a unit, whose inverse is its conjugate
            if not self.is_unit:
                raise ValueError("Negative powers of a non-unit need a modulus (its inverse isn't a Hurwitz integer)")

            base = self.conjugate()
            e = -e

        result = hurwitzint(1, 0, 0, 0)  # multiplicative identity
        while e:
            if e & 1:
                result *= base

            e >>= 1
            if e:
                base *= base

        return result

    # TODO: This only works around a mypyc bug, and can go once that is fixed. When its type check rejects an argument,
    #   mypyc's compiled __pow__ calls the right operand's __rpow__ itself, and without this, a hurwitzint's is the
    #   wrapper CPython adds for the slot __pow__ fills, which calls straight back. So x ** x, 2 ** x and None ** x
    #   recursed until RecursionError, rather than raising TypeError. TestPow.test_hurwitzint_exponents checks them.
    def __rpow__(self, other: object) -> object:
        # Nothing takes a hurwitzint as an exponent. -> object, since mypyc would cast NotImplemented to a hurwitzint
        return NotImplemented

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
    def __divmod__(self, other: hurwitzint | int | float) -> tuple[hurwitzint, hurwitzint]:
        """
        Nearest-lattice division in the Hurwitz quaternion order.

        We define quotient on the LEFT (Python-style):
            self = q * other + r

        Because multiplication is non-commutative, this is a specific choice.

        r is the remainder of least norm (at most half of abs(other)), and when several q leave that, the one with
            the largest numerator tuple. So r only depends on self modulo the left multiples of other: x and
            x + h*other leave the same remainder, and so do other and u*other for a unit u. For an integer other, that
            makes self % other the residue that pow(self, e, other) and inv_mod give.

        Returns:
            (q, r), or NotImplemented if other is an unsupported type.

        Raises:
            ZeroDivisionError: if other == 0
        """
        if _is_number(other):
            other = hurwitzint(other)

        if not isinstance(other, hurwitzint):
            return NotImplemented

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        # q ~ self * conj(other) / N(other), the usual quaternion self / other (see _division)
        return self._division(other, n)

    def __rdivmod__(self, other: int | float) -> tuple[hurwitzint, hurwitzint]:
        if _is_number(other):
            return hurwitzint(other).__divmod__(self)

        return NotImplemented

    def __truediv__(self, other: hurwitzint | int | float) -> hurwitzint:
        # mirror QuadInt: treat / as Euclidean division in this domain
        return self.__floordiv__(other)

    def __rtruediv__(self, other: int | float) -> hurwitzint:
        if _is_number(other):
            return hurwitzint(other).__truediv__(self)

        return NotImplemented

    def __floordiv__(self, other: hurwitzint | int | float) -> hurwitzint:
        # Not divmod(self, other), so an unsupported other gets its own __rfloordiv__ and a TypeError naming //
        qr = self.__divmod__(other)
        if qr is NotImplemented:
            return NotImplemented

        return qr[0]

    def __rfloordiv__(self, other: int | float) -> hurwitzint:
        if _is_number(other):
            return hurwitzint(other).__floordiv__(self)

        return NotImplemented

    def __mod__(self, other: hurwitzint | int | float) -> hurwitzint:
        qr = self.__divmod__(other)
        if qr is NotImplemented:
            return NotImplemented

        return qr[1]

    def __rmod__(self, other: int | float) -> hurwitzint:
        if _is_number(other):
            return hurwitzint(other).__mod__(self)

        return NotImplemented
    # endregion

    # region Right-division helpers
    def rdivmod(self, other: hurwitzint | int | float) -> tuple[hurwitzint, hurwitzint]:
        """
        Right-quotient division in the Hurwitz quaternion order.

        Defines quotient on the RIGHT:
            self = other * q + r

        r is chosen like the remainder of divmod (least norm, then the largest numerator tuple), so it only depends on
            self modulo the right multiples of other: x and x + other*h leave the same remainder, and so do other and
            other*u for a unit u.

        Returns:
            (q, r)

        Raises:
            TypeError: If other is an unsupported type.
            ZeroDivisionError: If trying to divide by 0.
        """
        if _is_number(other):
            other = hurwitzint(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for rdivmod: {type(other).__name__!r}")

        n = abs(other)
        if n == 0:
            raise ZeroDivisionError

        # Right quotient: q ~ other^{-1} * self = conj(other) * self / N(other) (see _division)
        return self._division(other, n, right=True)

    def rtruediv(self, other: hurwitzint | int | float) -> hurwitzint:
        """A version of __truediv__ for right-division"""
        return self.rfloordiv(other)

    def rfloordiv(self, other: hurwitzint | int | float) -> hurwitzint:
        """A version of __floordiv__ for right-division"""
        q, _ = self.rdivmod(other)
        return q

    def rmod(self, other: hurwitzint | int | float) -> hurwitzint:
        """A version of __mod__ for right-division"""
        _, r = self.rdivmod(other)
        return r
    # endregion
    # endregion

    # region Exact division
    def _exact_division(self, divisor: hurwitzint, *, right: bool = False) -> hurwitzint | None:
        """
        The exact division shared by exact_div_right and exact_div_left, and through _divides by divides_*.

        The quotient self / divisor is self * conj(divisor) / N(divisor) (or conj(divisor) * self / N(divisor)
            with right=True), as in _divmod_numerators, so each numerator of self * conj(divisor), over N(divisor),
            is a numerator of the quotient. That is a Hurwitz integer only if all four divide exactly,
            into four numerators of the same parity.

        Args:
            divisor: The divisor.
            right: Divide on the right (self == divisor*q), rather than on the left (self == q*divisor).

        Returns:
            hurwitzint | None: The quotient q, or None if no Hurwitz integer q divides exactly.

        Raises:
            ZeroDivisionError: If divisor is 0.
        """
        n = abs(divisor)
        if n == 0:
            raise ZeroDivisionError

        A, B, C, D = self._a, self._b, self._c, self._d
        E, F, G, H = divisor._a, divisor._b, divisor._c, divisor._d

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

    def exact_div_right(self, other: hurwitzint | int | float) -> hurwitzint | None:
        """
        Return q if self == q * other exactly, else None.

        This divides other off the right of self, when other is a right divisor of self: like gcd_right,
            it is named for the side of self that other divides. It is the exact version of divmod(self, other),
            whose quotient it returns whenever the remainder is 0, and it returns None otherwise. Like divmod,
            it raises ZeroDivisionError for an other of 0, and TypeError for anything but a hurwitzint, int or float.

        Returns:
            hurwitzint | None: The quotient q, or None if no Hurwitz integer q gives self == q * other.
        """
        return self._exact_division(_to_hurwitzint(other, "exact_div_right"))

    def exact_div_left(self, other: hurwitzint | int | float) -> hurwitzint | None:
        """
        Return q if self == other * q exactly, else None.

        This divides other off the left of self, when other is a left divisor of self: like gcd_left,
            it is named for the side of self that other divides. It is the exact version of self.rdivmod(other),
            whose quotient it returns whenever the remainder is 0, and it returns None otherwise. Like rdivmod,
            it raises ZeroDivisionError for an other of 0, and TypeError for anything but a hurwitzint, int or float.

        Returns:
            hurwitzint | None: The quotient q, or None if no Hurwitz integer q gives self == other * q.
        """
        return self._exact_division(_to_hurwitzint(other, "exact_div_left"), right=True)

    def _divides(self, dividend: hurwitzint, *, right: bool = False) -> bool:
        """
        The divisibility test shared by divides_right and divides_left: whether dividend divides exactly by self.

        Args:
            dividend: What self might divide.
            right: Whether dividend == self*q for some q, rather than dividend == q*self.

        Returns:
            bool: Whether self divides dividend on that side.
        """
        # Everything divides 0, and 0 divides only 0, so this never divides by 0
        if not self:
            return not dividend

        return dividend._exact_division(self, right=right) is not None

    def divides_right(self, other: hurwitzint | int | float) -> bool:
        """
        Return True iff self is a right divisor of other: other == q * self for some Hurwitz integer q.

        That is, iff other.exact_div_right(self) finds a quotient. Note which way round that is: self is the
            divisor, as in "self divides other". Everything divides 0, and 0 divides only 0, so unlike
            exact_div_right this never raises ZeroDivisionError. It does raise TypeError for an other that is not
            a hurwitzint, int or float.

        Returns:
            bool: Whether self divides other on the right.
        """
        return self._divides(_to_hurwitzint(other, "divides_right"))

    def divides_left(self, other: hurwitzint | int | float) -> bool:
        """
        Return True iff self is a left divisor of other: other == self * q for some Hurwitz integer q.

        That is, iff other.exact_div_left(self) finds a quotient. Note which way round that is: self is the
            divisor, as in "self divides other". Everything divides 0, and 0 divides only 0, so unlike
            exact_div_left this never raises ZeroDivisionError. It does raise TypeError for an other that is not
            a hurwitzint, int or float.

        Returns:
            bool: Whether self divides other on the left.
        """
        return self._divides(_to_hurwitzint(other, "divides_left"), right=True)
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

    # TODO: mypyc doesn't put __index__ in the type's nb_index slot yet (see AS_NUMBER_SLOT_DEFS in its emitclass.py),
    #   so compiled, only int() works (through __int__), and operator.index, range(x), a[x] and the like raise
    #   TypeError. TestConversions.test_index_where_python_wants_an_int will fail once it does, as a reminder.
    def __index__(self) -> int:
        """
        Return the int that self equals, for a real hurwitzint, so it works where Python wants an int, like range().

        Returns:
            int: The integer that self is.

        Raises:
            TypeError: If self has an i, j or k part, so that no int equals it.
        """
        if self._b or self._c or self._d:
            raise TypeError(f"cannot convert {self!r} to int, since it is not real")

        return self._a // 2

    def __int__(self) -> int:
        # Just __index__, as in quadint. int() would fall back to __index__ anyway, except that mypyc doesn't wire it up
        return self.__index__()

    def __float__(self) -> float:
        """
        Return the float that self equals, like int(self) but as a float. complex(self) goes through this too.

        Returns:
            float: The value of self, as a float.

        Raises:
            TypeError: If self has an i, j or k part, so that no float equals it.
        """
        if self._b or self._c or self._d:
            raise TypeError(f"cannot convert {self!r} to float, since it is not real")

        return float(self._a // 2)

    def __iter__(self) -> Iterator[int]:
        return iter((self._a, self._b, self._c, self._d))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> int:
        # The mypyc build rejects anything but an int before getting here (a slice too), so this makes pure Python
        #   match it, rather than answer x[1.0] with the i part, or call a slice out of range
        if not isinstance(idx, int):
            raise TypeError(f"hurwitzint indices must be integers, not {type(idx).__name__}")

        # A negative index counts back from the end, as with a tuple
        if idx < 0:
            idx += 4

        if idx == 0:
            return self._a
        if idx == 1:
            return self._b
        if idx == 2:
            return self._c
        if idx == 3:
            return self._d
        raise IndexError("hurwitzint index out of range (valid: -4..3)")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, hurwitzint):
            # Part by part, since mypyc compiles a tuple comparison into a slow generic one
            return self._a == other._a and self._b == other._b and self._c == other._c and self._d == other._d

        # Python numbers compare exactly (unlike arithmetic, which truncates floats with int()), since anything equal
        #   has to hash the same too (see __hash__). A complex off the real axis is never equal.
        if isinstance(other, int):
            return self._b == 0 and self._c == 0 and self._d == 0 and self._a == 2 * other

        # Separate checks, since mypyc compiles isinstance against a tuple to a generic call, but against float alone
        #   to a quick type check. (complex has no quick check, but only what isn't a float gets that far.)
        if isinstance(other, float):
            if not other.is_integer():
                return False

            return self._b == 0 and self._c == 0 and self._d == 0 and self._a == 2 * int(other)

        if isinstance(other, complex):
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
        #   rebuild neither a class with __slots__ (pure Python) nor a mypyc native class.
        # Every pickle names hurwitzint._make, so renaming it after a release would break the pickles saved before
        return hurwitzint._make, (self._a, self._b, self._c, self._d)

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

    @property
    def is_irreducible(self) -> bool:
        """
        Is this irreducible, a Hurwitz prime? That is, not 0 or a unit, and not a product of two non-units.

        That is exactly when its norm is a rational prime. Then in any product x = a*b, N(a)*N(b) is prime, so a or b
            has norm 1 and is a unit. Otherwise x is 0, a unit, or a product of Hurwitz primes, at least two of them
            (see factor_right). So a rational prime p is never irreducible here, being conj(P) * P for a P of norm p.

        Returns:
            bool: Whether self is irreducible.
        """
        return isprime(abs(self))

    def inverse(self) -> hurwitzint:
        """Find the inverse of the current hurwitzint (only applies to units)"""
        if not self.is_unit:
            raise ValueError("only Hurwitz units have inverses in the Hurwitz order")

        return self.conjugate()

    def __invert__(self) -> hurwitzint:
        """~u is the inverse of a unit u, like u.inverse(): its conjugate. Anything else raises ValueError."""
        return self.inverse()

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
             other: hurwitzint | int | float,
             *,
             right: bool = False) -> hurwitzint:
        """
        GCD via Euclidean algorithm, as whichever associate the algorithm lands on.

        This divides on the left (a = q*b + r) for a right gcd, or with right=True on the right (a = b*q + r)
            for a left gcd. The loop runs on plain int numerators (see _divmod_numerators), so the only
            hurwitzint it builds is the answer. Any remainder of least norm will do, so when several tie, it skips
            the work of picking the one % would (canonical=False).

        Returns:
            hurwitzint: The gcd.

        Raises:
            TypeError: If other is an unsupported type.
        """
        if _is_number(other):
            other = hurwitzint(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"Unable to divide hurwitzint and type {type(other)}")

        if not self:
            return other

        A, B, C, D = self._a, self._b, self._c, self._d
        E, F, G, H = other._a, other._b, other._c, other._d
        while E or F or G or H:
            n = (_mul(E, E) + _mul(F, F) + _mul(G, G) + _mul(H, H)) >> 2
            _, _, _, _, Ra, Rb, Rc, Rd = _divmod_numerators(A, B, C, D, E, F, G, H, n, right=right, canonical=False)
            A, B, C, D, E, F, G, H = E, F, G, H, Ra, Rb, Rc, Rd

        return self._make(A, B, C, D)

    def _xgcd(self,
              other: hurwitzint | int | float,
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
        if _is_number(other):
            other = hurwitzint(other)

        if not isinstance(other, hurwitzint):
            raise TypeError(f"unsupported type for xgcd: {type(other).__name__!r}")

        A, B, C, D = self._a, self._b, self._c, self._d
        E, F, G, H = other._a, other._b, other._c, other._d

        # The coefficients of A..D and of E..H, as numerators
        s0, t0 = (2, 0, 0, 0), (0, 0, 0, 0)
        s1, t1 = (0, 0, 0, 0), (2, 0, 0, 0)
        while E or F or G or H:
            n = (_mul(E, E) + _mul(F, F) + _mul(G, G) + _mul(H, H)) >> 2
            Qa, Qb, Qc, Qd, Ra, Rb, Rc, Rd = _divmod_numerators(A, B, C, D, E, F, G, H, n, right=right, canonical=False)
            q = (Qa, Qb, Qc, Qd)

            A, B, C, D, E, F, G, H = E, F, G, H, Ra, Rb, Rc, Rd
            s0, s1 = s1, _sub_mul_numerators(s0, q, s1, right=right)
            t0, t1 = t1, _sub_mul_numerators(t0, q, t1, right=right)

        sa, sb, sc, sd = s0
        ta, tb, tc, td = t0
        return self._make(A, B, C, D), self._make(sa, sb, sc, sd), self._make(ta, tb, tc, td)

    def gcd_right(self,
                  other: hurwitzint | int | float,
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
                 other: hurwitzint | int | float,
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
                   other: hurwitzint | int | float,
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
                  other: hurwitzint | int | float,
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

    # region Modular arithmetic
    @staticmethod
    def _modulus(mod: hurwitzint | int | float, name: str) -> int:
        """
        Return the positive integer that mod stands for, as the modulus of inv_mod or pow(x, e, mod).

        Only an integer works as a modulus. It commutes with everything, so its multiples are the same from either side
            (a two-sided ideal), and reducing modulo it works for products, on either side. The multiples of a
            hurwitzint off the real axis are not, so one of those raises ValueError. m and -m have the same multiples.

        Returns:
            int: abs(mod), with a float truncated with int() first, like everywhere else a float meets a hurwitzint.

        Raises:
            TypeError: If mod is not a hurwitzint, int or float.
            ValueError: If mod is a hurwitzint off the real axis.
            ZeroDivisionError: If mod is 0.
        """
        if isinstance(mod, hurwitzint):
            if mod._b or mod._c or mod._d:
                raise ValueError(f"{name} needs an integer modulus, not {mod!r}")

            m = mod._a // 2
        elif _is_number(mod):
            m = mod if type(mod) is int else int(mod)
        else:
            # The mypyc build rejects anything else before getting here, so this makes pure Python match it
            raise TypeError(f"unsupported type for {name}: {type(mod).__name__!r}")

        if m == 0:
            raise ZeroDivisionError(f"{name} modulus cannot be 0")

        return -m if m < 0 else m

    def inv_mod(self, mod: hurwitzint | int | float) -> hurwitzint:
        """
        Return the inverse of self modulo the integer mod: the y with x*y and y*x both congruent to 1 modulo mod.

        x * conj(x) == conj(x) * x == N(x), so if N(x) has an inverse k modulo mod, conj(x) * k is an inverse of x,
            on both sides. Otherwise x has none, since N(x)*N(y) == N(x*y) would have to be 1 modulo mod. So x is
            invertible modulo mod exactly when N(x) and mod are coprime, and then its inverse is unique modulo mod.

        It comes back as y % mod for any inverse y, like the result of pow(x, e, mod): the canonical residue of its
            class, the one of least norm with ties going to the largest numerator tuple.

        The modulus has to be an integer, since only an integer's multiples are the same on either side: a float is
            truncated with int() and a hurwitzint has to be real (or this raises ValueError). A modulus of 0 raises
            ZeroDivisionError, and anything but a hurwitzint, int or float raises TypeError.

        Returns:
            hurwitzint: The inverse, reduced modulo mod.

        Raises:
            ValueError: If self is not invertible modulo mod, which is when N(self) shares a factor with it.
        """
        m = self._modulus(mod, "inv_mod")
        n = abs(self)
        if gcd(n, m) != 1:
            raise ValueError(f"{self!r} is not invertible mod {m}, since its norm {n} shares a factor with it")

        # conj(self) * k, as numerators
        k = pow(n, -1, m)
        Ca, Cb, Cc, Cd = _mul(self._a, k), -_mul(self._b, k), -_mul(self._c, k), -_mul(self._d, k)
        Ra, Rb, Rc, Rd = _residue_numerators(Ca, Cb, Cc, Cd, m)
        return self._make(Ra, Rb, Rc, Rd)

    def _pow_mod(self, e: int, m: int) -> hurwitzint:
        """
        Return self**e modulo m > 0, as its canonical residue, for pow(self, e, m). A negative e uses inv_mod.

        m is an integer, so its multiples are a two-sided ideal, and the residues of a product only depend on those of
            its factors. So this reduces after every product, which keeps the numbers small (as Python's pow does
            for an int). It works on plain int numerators, like the division code, so it builds no hurwitzint until the
            answer.

        Returns:
            hurwitzint: The power, reduced.
        """
        base = self.inv_mod(m) if e < 0 else self
        if e < 0:
            e = -e

        Ra, Rb, Rc, Rd = _residue_numerators(2, 0, 0, 0, m)  # 1, which is 0 modulo 1
        Ba, Bb, Bc, Bd = _residue_numerators(base._a, base._b, base._c, base._d, m)

        # Exponentiation by squaring. Powers of one base commute, so the order of each product doesn't matter
        while e:
            if e & 1:
                Pa, Pb, Pc, Pd = _mul_numerators(Ra, Rb, Rc, Rd, Ba, Bb, Bc, Bd)
                Ra, Rb, Rc, Rd = _residue_numerators(Pa, Pb, Pc, Pd, m)

            e >>= 1
            if e:
                Pa, Pb, Pc, Pd = _mul_numerators(Ba, Bb, Bc, Bd, Ba, Bb, Bc, Bd)
                Ba, Bb, Bc, Bd = _residue_numerators(Pa, Pb, Pc, Pd, m)

        return self._make(Ra, Rb, Rc, Rd)
    # endregion

    # region Factoring
    def content(self) -> int:
        """
        Largest positive integer m such that self = m*q' with q' still a Hurwitz integer (0 for self = 0).

        In numerator units, m has to divide all four numerators, leaving four of the same parity. So m divides their
            gcd g, and g itself works unless the numerators over g come out of mixed parity. That only happens when all
            four numerators are even (when they are all odd, so are they over g), and then g is even and g/2 works,
            leaving four even numerators.

        Returns:
            int: Computed content value.
        """
        A, B, C, D = self._a, self._b, self._c, self._d
        g = gcd(A, B, C, D)  # Never negative, whatever the signs, and 0 only for self = 0
        if g == 0:
            return 0

        a, b, c, d = A // g, B // g, C // g, D // g
        if ((a ^ b) & 1) or ((a ^ c) & 1) or ((a ^ d) & 1):
            return g // 2

        return g

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

    @classmethod
    def prime_of_norm(cls, p: int | float, *, direction: Literal["left", "right"] = "right") -> hurwitzint:
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
        if not _is_number(p):
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

    def _factor_detail(self, direction: Literal["left", "right"]) -> NonCommutativeFactorization:
        """
        The factorization shared by factor_right_detail and factor_left_detail, which differ in one thing: the side.

        direction="right" peels each prime off the right of what is left, as its canonical right divisor of that norm
            (gcd_right), dividing it off with divmod. direction="left" peels it off the left instead, as its canonical
            left divisor (gcd_left), dividing it off with rdivmod.

        Returns:
            NonCommutativeFactorization: The factorization.

        Raises:
            ArithmeticError: If there is an unexpected problem preventing factoring, indicating a bug in the code.
        """
        if not self:
            return NonCommutativeFactorization(content=0, unit=hurwitzint(1, 0, 0, 0), primes=(), direction=direction)

        # The content is an integer, which commutes with everything, so it divides out the same from either side
        m = self.content()
        q = self // m if m > 1 else self

        # Now q is primitive, so it has one right divisor (and one left divisor) of each prime norm p dividing N(q),
        #   up to a unit, which is its gcd with p
        right = direction == "right"
        nf = _factorint(abs(q))

        primes: list[hurwitzint] = []
        # Largest norm first, since the primes come off the end and get reversed below
        for p in sorted(nf.keys(), reverse=True):
            for _ in range(nf[p]):
                pi = q.gcd_right(p) if right else q.gcd_left(p)
                if abs(pi) != p:
                    raise ArithmeticError(f"Failed to extract {direction} prime for {p=}")

                # q = qq * pi on the right, or q = pi * qq on the left
                qq, rr = divmod(q, pi) if right else q.rdivmod(pi)
                if rr:
                    raise ArithmeticError(f"extracted {direction} prime did not actually divide (unexpected)")

                q = qq
                primes.append(pi)

        # Remaining q must be a unit (norm 1) if we extracted all prime norms.
        if abs(q) != 1:
            raise ArithmeticError("remaining cofactor is not a unit; factorization incomplete")

        return NonCommutativeFactorization(content=m, unit=q, primes=tuple(reversed(primes)), direction=direction)

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
        """
        return self._factor_detail("right")

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
        """
        return self._factor_detail("left")

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


def _to_hurwitzint(n: hurwitzint | int | float, name: str) -> hurwitzint:
    """
    Return n as a hurwitzint, for the methods and module-level helpers that take a plain number in place of one.

    Those raise TypeError for anything else, naming themselves (unlike the operators, which return NotImplemented).
        So exact_div_right(3) divides by hurwitzint(3), and the module-level gcd_right(12, b) is
        hurwitzint(12).gcd_right(b).

    Returns:
        hurwitzint: n itself, or the hurwitzint equal to it (a float is truncated with int(), like everywhere else).

    Raises:
        TypeError: If n is not a hurwitzint, int or float.
    """
    if isinstance(n, hurwitzint):
        return n

    # The mypyc build rejects anything else before getting here, so this makes pure Python match it
    if not _is_number(n):
        raise TypeError(f"unsupported type for {name}: {type(n).__name__!r}")

    return hurwitzint(n)


def rdivmod(a: hurwitzint | int | float, b: hurwitzint | int | float) -> tuple[hurwitzint, hurwitzint]:
    """Simply a helper method to match existing Python divmod syntax, for a.rdivmod(b), where a can be a number too"""
    return _to_hurwitzint(a, "rdivmod").rdivmod(b)


def gcd_left(a: hurwitzint | int | float, b: hurwitzint | int | float) -> hurwitzint:
    """Simply a helper method to match existing Python gcd syntax, for a.gcd_left(b), where a can be a number too"""
    return _to_hurwitzint(a, "gcd_left").gcd_left(b)


def gcd_right(a: hurwitzint | int | float, b: hurwitzint | int | float) -> hurwitzint:
    """Simply a helper method to match existing Python gcd syntax, for a.gcd_right(b), where a can be a number too"""
    return _to_hurwitzint(a, "gcd_right").gcd_right(b)


def xgcd_left(a: hurwitzint | int | float, b: hurwitzint | int | float) -> tuple[hurwitzint, hurwitzint, hurwitzint]:
    """Simply a helper method to match gcd_left, for a.xgcd_left(b), where a can be a number too"""
    return _to_hurwitzint(a, "xgcd_left").xgcd_left(b)


def xgcd_right(a: hurwitzint | int | float, b: hurwitzint | int | float) -> tuple[hurwitzint, hurwitzint, hurwitzint]:
    """Simply a helper method to match gcd_right, for a.xgcd_right(b), where a can be a number too"""
    return _to_hurwitzint(a, "xgcd_right").xgcd_right(b)


def prod_right(x: Iterable[hurwitzint | int | float], start: hurwitzint | int | float | None = None):
    """Simply a helper method to match existing Python prod syntax"""
    if start is None:
        start = 1

    return prod(x, start=start)


def prod_left(x: Iterable[hurwitzint | int | float], start: hurwitzint | int | float | None = None):
    """Simply a helper method to match existing Python prod syntax"""
    if start is None:
        start = 1

    for sub_x in x:
        start = sub_x * start

    return start
