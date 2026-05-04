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
- **Reduced norm** $N(q) ∈ Z$ and quaternion conjugation
- **Euclidean division** (norm-Euclidean) via:
  - `divmod(a, b)` for **left-quotient** division (`a = q*b + r`)
  - `a.rdivmod(b)` (or `quatint.rdivmod(a, b)`) for **right-quotient** division (`a = b*q + r`)
- **Left/right gcd** (`gcd_left`, `gcd_right`) built on the corresponding division
- **Deterministic factorization** into `content`, `unit`, and Hurwitz primes (by prime norms)

New helper methods on every Hurwitz integer value:

* `x.content()` — largest positive integer `n` such that `x = n*y` for another Hurwitz integer `y`.
* `x.factor_right()` — a plain ordered tuple of factors whose product via `prod_right(...)` is exactly `x`.
* `x.factor_left()` — a plain ordered tuple of factors whose product via `prod_left(...)` is exactly `x`.
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

### Right-division (right-quotient)

Use `rdivmod` (method or helper) to define quotient on the **right**:

```python
from quatint import hurwitzint, rdivmod

a = hurwitzint(2, 3, 4, 53)
b = hurwitzint(1, 2, 3, 4)

q, r = rdivmod(a, b)
assert b * q + r == a
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

### Factorization

`quatint` exposes two levels of factorization API:

* `factor_right()` / `factor_left()` return a simple ordered tuple of factors.
* `factor_right_detail()` / `factor_left_detail()` return a structured `NonCommutativeFactorization` with metadata.

Use the plain methods when you just want factors that multiply back to the original value. 
  Use the detailed methods when you care about the separated integer content, unit, normalized prime factors, or canonical ordering.

### Plain factorization

The flat factorization methods return a tuple of `hurwitzint` factors. Because multiplication is non-commutative, the order and direction matter.

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

* `content`: maximal positive integer scalar dividing the element (in the Hurwitz sense)
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

## Representation & guarantees

### Internal representation

Values are stored in **numerator units**:

> `(A + B i + C j + D k) / 2`

This means:

* Lipschitz integers are stored with **even** numerators.
* True Hurwitz half-integers are stored with **odd** numerators.
* The constructor enforces the parity constraint.

### Scalar coercions

`int` and `float` inputs are accepted as scalars and converted via `int(...)` (i.e., truncation semantics). Quaternion values themselves remain exact.

## Public API (high level)

* `hurwitzint(a=0, b=0, c=0, d=0, *, half=False)`
* `hurwitzint.conjugate()`
* `abs(h)` → reduced norm `N(h)` (an `int`)
* `divmod(a, b)` → left-quotient Euclidean division
* `a.rdivmod(b)` / `rdivmod(a, b)` → right-quotient Euclidean division
* `a.gcd_left(b)` / `gcd_left(a, b)`
* `a.gcd_right(b)` / `gcd_right(a, b)`
* `a.factor_left_detail()` / `a.factor_right_detail()` → `NonCommutativeFactorization`
* `NonCommutativeFactorization.prod_left()` / `.prod_right()` / `.prod()`
