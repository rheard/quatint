# quatint

Exact (integer-backed) quaternion arithmetic for the **Hurwitz integers**.

`quatint` provides a fast, mypyc-friendly `hurwitzint` type that behaves like a small, practical numeric object: addition/subtraction/multiplication/power, norms and conjugation, plus **left/right Euclidean division**, **left/right gcd**, and **deterministic factorization** utilities.

## Why this exists

Python’s built-in numeric types don’t provide an exact, integer-backed quaternion type—especially not one that can represent the Hurwitz order `(a + b i + c j + d k) / 2` without floating point.

`quatint` keeps everything as integers under the hood while still letting you work with both:
- **Lipschitz integers**: $a, b, c, d ∈ Z$
- **Hurwitz “half” integers**: $(a + b i + c j + d k)/2$ with a parity constraint

## Key features

- **Exact arithmetic** (no floats required for quaternion values)
- **Hurwitz order representation** with parity enforcement
- **Non-commutative multiplication**
- **Reduced norm** $N(q) ∈ Z$, **trace** (`q + q.conjugate()`), and quaternion conjugation
- **Euclidean division** (norm-Euclidean) via:
  - `divmod(a, b)` for **left-quotient** division (`a = q*b + r`)
  - `a.rdivmod(b)` (or `quatint.rdivmod(a, b)`) for **right-quotient** division (`a = b*q + r`)
- **Exact division** (`exact_div_right`, `exact_div_left`), which gives `None` rather than leave a remainder,
  and **divisibility** tests (`divides_right`, `divides_left`)
- **Left/right gcd** (`gcd_left`, `gcd_right`) built on the corresponding division,
  and **extended gcd** (`xgcd_left`, `xgcd_right`) with Bézout coefficients
- **Deterministic factorization** into `content`, `unit`, and Hurwitz primes (by prime norms)

New helper methods on every Hurwitz integer value:

* `x.content()` — largest positive integer `n` such that `x = n*y` for another Hurwitz integer `y`.
* `x.factor_right()` — a plain ordered tuple of Hurwitz primes whose product via `prod_right(...)` is exactly `x`.
* `x.factor_left()` — a plain ordered tuple of Hurwitz primes whose product via `prod_left(...)` is exactly `x`.
* `x.factor_right_detail()` — structured right factorization as a `NonCommutativeFactorization`.
* `x.factor_left_detail()` — structured left factorization as a `NonCommutativeFactorization`.

## Installation

```bash
python -m pip install quatint
````

This project is designed to compile cleanly with **mypyc** for speed (CI/test setups often ensure the compiled artifact is what’s running).

## Quick start

```python
from quatint import hurwitzint

a = hurwitzint(1, 1, 1, 1)
b = hurwitzint(2, 3, 4, 5)

print(a)         # (1+i+j+k)
print(a * b)     # (-10+6i+4j+8k)
print(b * a)     # (-10+4i+8j+6k)
```

### Half-integers (Hurwitz elements)

Use `half=True` to provide numerator components of a `/2` element:

```python
from quatint import hurwitzint

h = hurwitzint(1, 3, 5, 7, half=True)
print(h)  # (1+3i+5j+7k)/2
```

### Division (left-quotient)

`divmod(a, b)` defines quotient on the **left**:

```python
from quatint import hurwitzint

a = hurwitzint(2, 3, 4, 53)
b = hurwitzint(1, 2, 3, 4)

q, r = divmod(a, b)
assert q * b + r == a
```

The remainder is the one of least norm (at most half of `abs(b)`), and when several quotients leave that, the one with the largest numerator tuple.
    So it only depends on `a` modulo the multiples of `b`: `a % b == (a + h * b) % b` for any Hurwitz integer `h`.

### Right-division (right-quotient)

Use `rdivmod` (method or helper) to define quotient on the **right**:

```python
from quatint import hurwitzint, rdivmod

a = hurwitzint(2, 3, 4, 53)
b = hurwitzint(1, 2, 3, 4)

q, r = rdivmod(a, b)
assert b * q + r == a
```

### Exact division and divisibility

`divmod` and `rdivmod` always find a quotient, and leave whatever remainder they have to.
    `exact_div_right` and `exact_div_left` only divide when nothing would be left over, and return `None` otherwise:

```python
from quatint import hurwitzint

y = hurwitzint(1, 1, 1, 0)      # 1+i+j
x = hurwitzint(0, 1, 0, 0) * y  # i*y

assert x.exact_div_right(y) == hurwitzint(0, 1, 0, 0)  # x == i * y
assert x.exact_div_left(y) is None                     # but no q gives x == y * q
```

Like `gcd_right` and `gcd_left`, they are named for the side of `x` that `y` divides:
    `x.exact_div_right(y)` is the exact version of `divmod(x, y)` (`x == q*y`),
    and `x.exact_div_left(y)` is the exact version of `rdivmod(x, y)` (`x == y*q`).

To just ask whether `y` divides `x`, use `y.divides_right(x)` or `y.divides_left(x)` (`y` goes first, as in "`y` divides `x`").
    Everything divides `0`, and `0` divides only `0`, so these never raise `ZeroDivisionError`:

```python
from quatint import hurwitzint

y = hurwitzint(1, 1, 1, 0)      # 1+i+j
x = hurwitzint(0, 1, 0, 0) * y  # i*y

assert y.divides_right(x) and not y.divides_left(x)
assert hurwitzint(1, 1, 0, 0).divides_left(2)  # 2 == (1+i) * (1-i)
assert y.divides_right(0) and not hurwitzint(0).divides_right(x)
```

### GCD (left and right)

Because multiplication is non-commutative, there are *two* natural gcd notions:

```python
from quatint import hurwitzint

a = hurwitzint(2, 3, 4, 53)
b = hurwitzint(1, 2, 3, 4)

gl = a.gcd_left(b)    # common left divisor (a = gl*x, b = gl*y)
gr = a.gcd_right(b)   # common right divisor (a = x*gr, b = y*gr)
```

A gcd is only unique up to a unit on one side, so both return a canonical choice: the associate with the largest real part
    (then the largest i, j and k parts, to break ties). The result doesn't depend on the argument order,
    and the gcd of two integers is their usual positive gcd:

```python
from quatint import hurwitzint

assert hurwitzint(-6).gcd_right(15) == 3
```

### Extended gcd

`xgcd_right` and `xgcd_left` return the same gcd `g`, along with Bézout coefficients `s` and `t` that make it out of the two arguments.
    A right gcd is a combination with coefficients on the left, and a left gcd one with coefficients on the right:

```python
from quatint import hurwitzint

a = hurwitzint(2, 3, 4, 53)
b = hurwitzint(1, 2, 3, 4)

g, s, t = a.xgcd_right(b)
assert g == a.gcd_right(b)
assert s * a + t * b == g

g, s, t = a.xgcd_left(b)
assert g == a.gcd_left(b)
assert a * s + b * t == g
```

Like `gcd_left` and `gcd_right`, they also come as functions: `xgcd_left(a, b)` and `xgcd_right(a, b)`.

### Modular arithmetic

`x.inv_mod(m)` is the inverse of `x` modulo an integer `m`, on both sides: `x * y` and `y * x` are both 1 modulo `m`.
    It exists when the norm `abs(x)` and `m` are coprime, and otherwise this raises `ValueError`:

```python
from quatint import hurwitzint

x = hurwitzint(1, 1, 0, 0)  # 1+i, of norm 2
y = x.inv_mod(3)
print(y)  # (-1+i+0j+0k)
assert hurwitzint(3).divides_right(x * y - 1) and hurwitzint(3).divides_right(y * x - 1)
```

`pow(x, e, m)` is `x ** e` modulo an integer `m`, reducing as it goes, so a huge `e` is no problem.
    A negative `e` takes powers of `x.inv_mod(m)`, as Python's `pow` does for an `int`:

```python
from quatint import hurwitzint

x = hurwitzint(1, 2, 3, 4)
assert pow(x, 10**18, 7) == pow(pow(x, 10**9, 7), 10**9, 7)
assert pow(x, -1, 7) == x.inv_mod(7)
```

For both, the modulus has to be an integer (or a real `hurwitzint`), since only an integer's multiples are the same from either side.
    The result is reduced with `%`, so `pow(x, e, m) == (x ** e) % m`, as for an `int`, and congruent values always give the very same result.

### Primes of a given norm

Every rational prime `p` is the norm of a Hurwitz prime, and `hurwitzint.prime_of_norm(p)` returns a fixed one (always the same one for the same `p`).
    A quaternion times its conjugate is its norm, so this also splits `p` into two Hurwitz primes:

```python
from quatint import hurwitzint

P = hurwitzint.prime_of_norm(5)
print(P)  # (2+0i-j+0k)
assert abs(P) == 5
assert P.conjugate() * P == 5
```

By default the prime is canonical up to a unit on its left, like the primes of `factor_right_detail()`.
    `direction="left"` gives one that is canonical up to a unit on its right, like those of `factor_left_detail()`.

### Factorization

`quatint` exposes two levels of factorization API:

* `factor_right()` / `factor_left()` return a simple ordered tuple of Hurwitz primes.
* `factor_right_detail()` / `factor_left_detail()` return a structured `NonCommutativeFactorization` with metadata.

Use the plain methods when you just want primes that multiply back to the original value. 
  Use the detailed methods when you care about the separated integer content, unit, normalized prime factors, or canonical ordering.

### Plain factorization

The flat factorization methods return a tuple of Hurwitz primes (each has a rational prime as its norm), with the unit folded into the first one.
    That includes the primes of the integer content, factored as described under [Factoring the content](#factoring-the-content).
    Because multiplication is non-commutative, the order and direction matter.

```python
from quatint import hurwitzint, prod_left, prod_right

n = hurwitzint(2, 3, 4, 53)

right_factors = n.factor_right()
assert prod_right(right_factors) == n

left_factors = n.factor_left()
assert prod_left(left_factors) == n
```

For `factor_right()`, multiply the returned factors using `prod_right(...)`, which behaves like ordinary left-to-right multiplication:

```python
assert prod_right((a, b, c)) == a * b * c
```

For `factor_left()`, multiply the returned factors using `prod_left(...)`, which multiplies each new factor on the left:

```python
assert prod_left((a, b, c)) == c * b * a
```

### Detailed factorization

The detailed methods return a compact normal form, in a class called `NonCommutativeFactorization` with these properties:

* `content`: maximal positive integer scalar dividing the element (in the Hurwitz sense), or 1 after `expand_content()`
* `unit`: a norm-1 Hurwitz unit (deterministically chosen)
* `primes`: Hurwitz primes (each with prime rational norm), normalized via unit migration

`NonCommutativeFactorization` also exposes some utility methods for convenience, such as:

```python
from quatint import hurwitzint

n = hurwitzint(2, 3, 4, 53)

fr = n.factor_right_detail()
assert fr.prod_right() == n

fl = n.factor_left_detail()
assert fl.prod_left() == n
```

#### Factoring the content

`content` stays an integer, since that is unique, and its factorization into Hurwitz primes is not: every rational prime `p` is `conj(P) * P` for a Hurwitz prime `P` of norm `p`,
    but `p + 1` of those `P` give genuinely different factorizations (just one for `p = 2`).
    `expand_content()` makes a fixed choice, `P = hurwitzint.prime_of_norm(p)`, and returns the factorization with those primes merged in by norm, and `content` set to 1:

```python
from quatint import hurwitzint

f = hurwitzint(527).factor_right_detail()  # 527 = 17 * 31
print(f.content, f.primes)  # 527 ()

f = f.expand_content()
print(f.content, [abs(p) for p in f.primes])  # 1 [17, 17, 31, 31]
assert f.prod_right() == 527
```

That means factoring `content` as an integer, which the detailed methods never do, and which can be slow for a big content with big prime factors.
    If you already know them, pass them in to skip that: `f.expand_content(factors={17: 1, 31: 1})`.

## Representation & guarantees

### Internal representation

Values are stored in **numerator units**:

> `(A + B i + C j + D k) / 2`

This means:

* Lipschitz integers are stored with **even** numerators.
* True Hurwitz half-integers are stored with **odd** numerators.
* The constructor enforces the parity constraint.

### Scalar coercions

`int` and `float` inputs are accepted as scalars and converted via `int(...)` (i.e., truncation semantics), both as operands and as the components (and exponents) given to `hurwitzint(...)` and `**`. Quaternion values themselves remain exact.

That includes either argument of the module-level `rdivmod`, `gcd_left`, `gcd_right`, `xgcd_left` and `xgcd_right`, so `gcd_right(12, b)` is `hurwitzint(12).gcd_right(b)`.

Equality is the exception, and is exact: `hurwitzint(2) == 2` and `hurwitzint(2) == 2.0` are `True`, but `hurwitzint(2) == 2.5` is `False`. A value that equals a Python number also hashes like it, so `hurwitzint(2)` and `2` are the same dict key (as `2` and `2.0` are).

## Public API (high level)

* `hurwitzint(a=0, b=0, c=0, d=0, *, half=False)`
* `hurwitzint.conjugate()`
* `~u` / `u.inverse()` → the inverse of a unit `u` (its conjugate), or `ValueError` for anything else
* `x ** n` → a power, where a negative `n` only works for a unit
* `pow(x, n, m)` → `x ** n` modulo the integer `m`, where a negative `n` takes powers of `x.inv_mod(m)`
* `abs(h)` → reduced norm `N(h)` (an `int`)
* `h.trace` → `h + h.conjugate()`, twice the real part (an `int`); every `h` is a root of `h**2 - h.trace*h + abs(h)`
* `h.is_irreducible` → whether `h` is a Hurwitz prime, which is when its norm is a rational prime
* `int(h)` / `float(h)` / `complex(h)` → the number a real `h` equals, or `TypeError` for anything else
* `divmod(a, b)` → left-quotient Euclidean division
* `a.rdivmod(b)` / `rdivmod(a, b)` → right-quotient Euclidean division
* `a.exact_div_right(b)` / `a.exact_div_left(b)` → the `q` with `a == q*b` / `a == b*q`, or `None` if there is none
* `b.divides_right(a)` / `b.divides_left(a)` → whether `a == q*b` / `a == b*q` for some `q`
* `a.gcd_left(b)` / `gcd_left(a, b)`
* `a.gcd_right(b)` / `gcd_right(a, b)`
* `a.xgcd_left(b)` / `xgcd_left(a, b)` → `(g, s, t)` with `a*s + b*t == g`
* `a.xgcd_right(b)` / `xgcd_right(a, b)` → `(g, s, t)` with `s*a + t*b == g`
* `a.inv_mod(m)` → the inverse of `a` modulo the integer `m`, on both sides
* `hurwitzint.prime_of_norm(p, *, direction="right")` → a fixed Hurwitz prime of norm `p`
* `a.factor_left_detail()` / `a.factor_right_detail()` → `NonCommutativeFactorization`
* `NonCommutativeFactorization.prod_left()` / `.prod_right()` / `.prod()`
* `NonCommutativeFactorization.expand_content(factors=None)` → the same factorization, with the content factored into Hurwitz primes too
