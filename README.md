This package provides a `hurwitzint` class for dealing with Hurwitz integers.

Float operations are avoided by this package for infinite precision. To input half-integers with infinite precision,
    use the `half` argument as so:

```python
from quatint import hurwitzint

a = hurwitzint(3, 5, 7, 11)

print(a)  # Outputs "3+5i+7j+11k"
```

## Examples

```python
from quatint import hurwitzint

a = hurwitzint(1, 2, 3, 4)
b = hurwitzint(2, 3, 4, 5)

c = a * b
print(c)  # Outputs "-36+6i+12j+12k"
```

## Disclaimer

This is intended for use with discrete mathematics, and ideally will be limited to the
    operations: add, sub, mul, and pow.

Trying to divide using this class, or using floats with this class, will (probably) result in integer conversion cutoff.

As an example of this problem, note the equivalences below:
```python
from quatint import hurwitzint

a = hurwitzint(1, 2, 3, 4)

print(a / 3)  # Outputs "i+j+k"
print(a / 3.5)  # Outputs "i+j+k"

print(a + 1)  # Outputs "2+2i+3j+4k"
print(a + 1.5)  # Outputs "2+2i+3j+4k"
```