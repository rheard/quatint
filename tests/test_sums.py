from __future__ import annotations

import math
import os
import random

from functools import cache
from pathlib import Path

import pytest

from pytest import mark
from sympy import nextprime, primerange

import quatint.sums

from quatint.quat import hurwitzint
from quatint.sums import (
    _canonical_lipschitz_tuple,
    _prime_uv_witnesses,
    _seed_candidates,
    _lipschitz_unit_orbit,
    decompose_number,
    decompose_prime,
)

@pytest.mark.skipif(os.getenv("CI", "").lower() not in {"1", "true", "yes"}, reason="Compiled-only test")
def test_compiled_tests():
    """Verify that we are running these tests with a compiled version of quatint.sums."""
    path = Path(quatint.sums.__file__)
    assert path.suffix.lower() != ".py"


def _sum4(sol: tuple[int, int, int, int]) -> int:
    """Return a^2 + b^2 + c^2 + d^2 for a four-square tuple."""
    a, b, c, d = sol
    return a * a + b * b + c * c + d * d


def _assert_canonical(sol: tuple[int, int, int, int]) -> None:
    """Validate canonical nonnegative sorted four-square coordinates."""
    assert len(sol) == 4
    assert tuple(sorted(sol)) == sol
    assert all(isinstance(x, int) for x in sol)
    assert all(x >= 0 for x in sol)


@cache
def _brute_force_foursquares(n: int) -> frozenset[tuple[int, int, int, int]]:
    """
    Brute-force canonical nonnegative four-square decompositions of n.

    Solutions are canonicalized as 0 <= a <= b <= c <= d.
    """
    out: set[tuple[int, int, int, int]] = set()
    lim = math.isqrt(n)

    for a in range(lim + 1):
        ra = n - a * a
        if ra < 0:
            break

        for b in range(a, math.isqrt(ra) + 1):
            rab = ra - b * b
            if rab < 0:
                break

            for c in range(b, math.isqrt(rab) + 1):
                rabc = rab - c * c
                if rabc < 0:
                    break

                d = math.isqrt(rabc)
                if d < c:
                    continue

                if d * d == rabc:
                    out.add((a, b, c, d))

    return frozenset(out)


def brute_force_foursquares(
    n: int,
    *,
    no_trivial_solutions: bool = False,
) -> set[tuple[int, int, int, int]]:
    """Brute-force all canonical four-square decompositions, optionally filtering zeros."""
    sols = set(_brute_force_foursquares(n))
    if no_trivial_solutions:
        sols = {sol for sol in sols if sol[0] != 0}
    return sols


class TestPrimeDecomposition:
    """Tests for decompose_prime."""

    @mark.parametrize("p", [2, 3, 5, 7, 13, 17, 29, 31, 97], ids=str)
    def test_prime_examples(self, p: int):
        """Verify hand-picked prime decompositions match brute force exactly."""
        got = decompose_prime(p)
        expect = brute_force_foursquares(p)

        assert got == expect, f"Mismatch for prime {p}: missing={expect - got}, extra={got - expect}"

    def test_primes_below_200(self):
        """Verify many small primes against brute force exactly."""
        for p in primerange(100, 200):
            got = decompose_prime(p)
            expect = brute_force_foursquares(p)

            assert got == expect, f"Mismatch for prime {p}: missing={expect - got}, extra={got - expect}"


# region Test direct lipschitz shortcut
def _direct_and_orbit_prime_solutions(p: int) -> tuple[
    set[tuple[int, int, int, int]],
    set[tuple[int, int, int, int]],
]:
    """
    Compare the direct-Lipschitz shortcut against full unit-orbit expansion.

    This specifically tests whether replacing:

        sols |= _lipschitz_unit_orbit(g)

    with:

        sol = _canonical_lipschitz_tuple(g)
        if sol is not None:
            sols.add(sol)

    loses any canonical four-square prime decompositions.
    """
    scalar = hurwitzint(p, 0, 0, 0)

    direct_sols: set[tuple[int, int, int, int]] = set()
    orbit_sols: set[tuple[int, int, int, int]] = set()

    for u, v in _prime_uv_witnesses(p):
        for candidate in _seed_candidates(u, v):
            for g in (
                candidate.gcd_right(scalar, normalize=False),
                candidate.gcd_left(scalar, normalize=False),
            ):
                if abs(g) != p:
                    continue

                direct = _canonical_lipschitz_tuple(g)
                if direct is not None:
                    direct_sols.add(direct)

                orbit_sols |= _lipschitz_unit_orbit(g)

    return direct_sols, orbit_sols


def test_direct_lipschitz_shortcut_matches_unit_orbit_for_random_high_primes():
    """Verify direct Lipschitz gcd outputs do not miss unit-orbit prime decompositions."""
    rng = random.Random(0)

    # uv_for_prime scans up to p, so keep this high-ish but not absurd.
    count = 1
    lower = 200
    upper = 500

    for _ in range(count):
        p = int(nextprime(rng.randrange(lower, upper)))

        direct_sols, orbit_sols = _direct_and_orbit_prime_solutions(p)

        assert direct_sols == orbit_sols, (
            f"Direct Lipschitz shortcut missed solutions for p={p}: "
            f"missing={orbit_sols - direct_sols}, extra={direct_sols - orbit_sols}"
        )

# endregion


class TestNumberDecomposition:
    """Tests for decompose_number."""

    @mark.parametrize(
        ("n", "expected"),
        [
            (1, {(0, 0, 0, 1)}),
            (2, {(0, 0, 1, 1)}),
            (3, {(0, 1, 1, 1)}),
            (4, {(0, 0, 0, 2), (1, 1, 1, 1)}),
            (5, {(0, 0, 1, 2)}),
            (20, {(0, 0, 2, 4), (1, 1, 3, 3)}),
        ],
        ids=str,
    )
    def test_examples(self, n: int, expected: set[tuple[int, int, int, int]]):
        """Verify small known decompositions."""
        assert decompose_number(n) == expected

    @mark.parametrize("no_trivial_solutions", [False, True], ids=str)
    def test_small_numbers_match_bruteforce(self, *, no_trivial_solutions: bool):
        """Verify all small numbers against brute force."""
        max_n = 150 if os.getenv("CI") else 400

        for n in range(1, max_n + 1):
            got = decompose_number(n, no_trivial_solutions=no_trivial_solutions)
            expect = brute_force_foursquares(n, no_trivial_solutions=no_trivial_solutions)

            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    @mark.parametrize(
        "n",
        [
            325,
            999,
            1_000,
            10_000,
            10_001,
        ],
        ids=str,
    )
    def test_larger_examples_match_bruteforce(self, n: int):
        """Verify selected larger examples against brute force."""
        got = decompose_number(n)
        expect = brute_force_foursquares(n)

        assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"

    def test_no_trivial_solutions_filters_zero_coordinate(self):
        """Verify no_trivial_solutions discards lower-dimensional solutions with a zero coordinate."""
        assert decompose_number(4, no_trivial_solutions=False) == {
            (0, 0, 0, 2),
            (1, 1, 1, 1),
        }
        assert decompose_number(4, no_trivial_solutions=True) == {
            (1, 1, 1, 1),
        }

        assert decompose_number(20, no_trivial_solutions=False) == {
            (0, 0, 2, 4),
            (1, 1, 3, 3),
        }
        assert decompose_number(20, no_trivial_solutions=True) == {
            (1, 1, 3, 3),
        }

    def test_prime_inputs_match_prime_decomposition(self):
        """Verify decompose_number and decompose_prime agree on prime inputs."""
        for p in primerange(2, 200):
            assert decompose_prime(p) == decompose_number(p)
            assert decompose_number(p) == brute_force_foursquares(p)

    def test_fuzzed_small_numbers_match_bruteforce(self):
        """Validate completeness for deterministic random examples."""
        rng = random.Random(0)
        count = 25 if os.getenv("CI") else 100

        for _ in range(count):
            n = rng.randrange(1, 2_000)
            got = decompose_number(n)
            expect = brute_force_foursquares(n)

            assert got == expect, f"Mismatch for n={n}: missing={expect - got}, extra={got - expect}"
