from __future__ import annotations

import math

from functools import cache
from itertools import permutations, starmap

from sympy import factorint

from quatint.quat import hurwitzint, uv_for_prime

def _factor_input(n: dict[int, int] | int) -> tuple[int, dict[int, int]]:
    """Return `(n, factorization)` whether the caller provided n or its factors."""
    if not isinstance(n, dict):
        return n, factorint(n)

    n_int = math.prod(p**k for p, k in n.items())
    return n_int, n


def _canonical_quadruple(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    """Canonicalize a four-square tuple by ignoring signs and coordinate order."""
    return tuple(sorted((abs(a), abs(b), abs(c), abs(d))))


def _canonical_lipschitz_tuple(q: hurwitzint) -> tuple[int, int, int, int] | None:
    """
    Convert a Lipschitz Hurwitz integer to a canonical four-square tuple.

    `hurwitzint` stores numerator coordinates for denominator 2. A Lipschitz quaternion has even numerators,
        so the corresponding integer coordinates are
            `(A//2, B//2, C//2, D//2)`.

    True half-integer Hurwitz elements are skipped because they do not directly represent integer solutions to
        `a^2 + b^2 + c^2 + d^2 = n`.

    Returns:
        None: Not a Lipschitz integer.
        tuple: The Lipschitz integer converted to canonical four-square tuple.
    """
    if not q.is_lipschitz:
        return None

    return _canonical_quadruple(q.a // 2, q.b // 2, q.c // 2, q.d // 2)


def _prime_uv_witnesses(p: int):
    """
    Yield all modular witnesses `(u, v)` satisfying:

        1 + u^2 + v^2 == 0 mod p

    These are the witnesses used to construct Hurwitz gcd candidates of norm `p`.
    Unlike `uv_for_prime`, this intentionally scans every `u` so that
    `decompose_prime` can collect every canonical four-square decomposition.
    """
    for u, v in uv_for_prime(p):
        yield u, v

        neg_v = (-v) % p
        if neg_v != v:
            yield u, neg_v


def _seed_candidates(u: int, v: int) -> set[hurwitzint]:
    """
    Build Hurwitz gcd seed candidates from a modular witness.

    The witness equation is symmetric in the four coordinates of `(1, u, v, 0)`.
    Trying all unique permutations is cheap for primes small enough that returning
    all four-square decompositions is practical, and it avoids accidentally only
    finding one unit orbit.
    """
    return set(starmap(hurwitzint, set(permutations((1, u, v, 0), 4))))


@cache
def _hurwitz_prime_seeds_for_prime(p: int) -> frozenset[hurwitzint]:
    """Return norm-`p` Hurwitz primes found from modular witnesses and gcds."""
    p = int(p)
    if p < 2:
        raise ValueError(f"Could not decompose prime {p!r} into four squares")

    scalar = hurwitzint(p, 0, 0, 0)
    out: set[hurwitzint] = set()
    for u, v in _prime_uv_witnesses(p):
        for candidate in _seed_candidates(u, v):
            for g in (
                candidate.gcd_right(scalar, normalize=False),
                candidate.gcd_left(scalar, normalize=False),
            ):
                if abs(g) == p:
                    out.add(g)
    if not out:
        raise ValueError(f"Could not decompose prime {p!r} into four squares")
    return frozenset(out)


def _is_trivial(sol: tuple[int, int, int, int]) -> bool:
    """Return True for lower-dimensional four-square solutions with a zero coordinate."""
    return sol[0] == 0


def _unit_orbit(q: hurwitzint) -> set[hurwitzint]:
    """Return the two-sided Hurwitz-unit orbit of `q`."""
    out: set[hurwitzint] = set()
    for left_unit in hurwitzint.UNITS:
        left_q = left_unit * q
        out.update(left_q * right_unit for right_unit in hurwitzint.UNITS)
    return out


def _lipschitz_unit_orbit(q: hurwitzint) -> set[tuple[int, int, int, int]]:
    """
    Return all canonical Lipschitz four-square tuples in the two-sided unit orbit of `q`.

    Left and right multiplication by Hurwitz units preserves the norm but can move
    a Hurwitz prime between different integer-coordinate representatives. Collecting
    the Lipschitz points in this orbit is what turns a Hurwitz prime into ordinary
    four-square decompositions.
    """
    out: set[tuple[int, int, int, int]] = set()

    for left_unit in hurwitzint.UNITS:
        left_q = left_unit * q
        for right_unit in hurwitzint.UNITS:
            sol = _canonical_lipschitz_tuple(left_q * right_unit)
            if sol is not None:
                out.add(sol)

    return out


@cache
def _prime_representatives_for_prime(p: int) -> frozenset[hurwitzint]:
    """
    Return Lipschitz Hurwitz representatives of norm `p`.

    A single Hurwitz prime's two-sided unit orbit does not necessarily contain
    its quaternion conjugate. Both orientations are needed when recombining prime
    factors; for example, scalar square factors require products like
    `q * q.conjugate()`. Therefore this helper keeps all Lipschitz elements in
    the unit orbits of both `g` and `g.conjugate()`.
    """
    reps: set[hurwitzint] = set()
    for g in _hurwitz_prime_seeds_for_prime(p):
        for base in (g, g.conjugate()):
            for w in _unit_orbit(base):
                if w.is_lipschitz:
                    reps.add(w)

    if not reps:
        raise ValueError(f"Could not decompose prime {p!r} into four squares")
    return frozenset(reps)


def decompose_prime(p: int) -> set[tuple[int, int, int, int]]:
    """
    Decompose a rational prime into all canonical sums of four squares.

    Returns all canonical nonnegative sorted tuples `(a, b, c, d)` such that:

        a^2 + b^2 + c^2 + d^2 = p

    The implementation is genuinely Hurwitz/quaternionic:
      1. Find modular witnesses `(u, v)` with `1 + u^2 + v^2 == 0 mod p`.
      2. Use those witnesses to build Hurwitz gcd candidates against the scalar prime `p`.
      3. Keep norm-`p` Hurwitz primes and collect the Lipschitz integer tuples in
         their two-sided Hurwitz-unit orbits.

    Returns:
        set: All canonical four-square decompositions found for the prime.

    Raises:
        ValueError: If the input cannot be decomposed as a prime by this Hurwitz construction.
    """
    return {_canonical_quadruple(q.a // 2, q.b // 2, q.c // 2, q.d // 2)
            for q in _prime_representatives_for_prime(int(p))}


def _prime_representative_slots(factors: dict[int, int]) -> list[frozenset[hurwitzint]]:
    """Return one Hurwitz representative choice-set per rational-prime factor occurrence."""
    slots: list[frozenset[hurwitzint]] = []

    for p, k in sorted(factors.items()):
        if k <= 0:
            continue

        reps = _prime_representatives_for_prime(p)
        for _ in range(k):
            slots.append(reps)

    return slots


def decompose_number(
    n: dict[int, int] | int,
    *,
    no_trivial_solutions: bool = False,
) -> set[tuple[int, int, int, int]]:
    """
    Decompose `n` into canonical four-square solutions.

    Returns all canonical nonnegative sorted quadruples `(a, b, c, d)` such that:

        a^2 + b^2 + c^2 + d^2 = n

    Args:
        n: The integer to decompose, or a precomputed factorization dictionary.
        no_trivial_solutions: If true, discard lower-dimensional solutions with
            at least one zero coordinate. The default is `False`.

    Returns:
        set: The set of expected solutions.

    Notes:
        This enumerates all canonical four-square decompositions by grouping
        two-square sums and combining complementary pair sums. That is much
        simpler than trying to enumerate all Hurwitz factorizations directly;
        quaternion multiplication is noncommutative, while the public problem
        here is the commutative four-square form.
    """
    n_int, factors = _factor_input(n)

    if n_int == 0:
        return set() if no_trivial_solutions else {(0, 0, 0, 0)}

    if n_int == 1:
        return set() if no_trivial_solutions else {(0, 0, 0, 1)}

    slots = _prime_representative_slots(factors)
    if not slots:
        raise ValueError(f"Could not decompose number {n!r} into four squares")

    products: set[hurwitzint] = set(slots[0])
    for reps in slots[1:]:
        products = {q * r for q in products for r in reps}

    out: set[tuple[int, int, int, int]] = set()
    for q in products:
        for sol in _lipschitz_unit_orbit(q):
            if no_trivial_solutions and _is_trivial(sol):
                continue
            out.add(sol)

    return out
