import pytest

from quatint.utils import cache_generator

class CacheGeneratorTests:
    """Support methods and state for cache_generator tests."""

    def setup_method(self, _):
        """Reset counters for each test."""
        self.created = 0
        self.yielded = 0

    def make_counting_range(self):
        """Create a cached generator that yields 0..n-1 and records creation/yield counts."""

        def _gen(n: int):
            self.created += 1
            for i in range(n):
                self.yielded += 1
                yield i

        return cache_generator(_gen)

    def make_error_range(self):
        """Create a cached generator that yields 0..(error_at-1) then raises RuntimeError."""

        def _gen(n: int, *, error_at: int):
            self.created += 1
            for i in range(n):
                if i == error_at:
                    raise RuntimeError("error occurred")
                self.yielded += 1
                yield i

        return cache_generator(_gen)


class TestBasics(CacheGeneratorTests):
    """Tests for basic caching and replay behavior."""

    def test_returns_fresh_generator_each_call(self):
        """Each call should return a fresh generator object (even for same signature)."""
        wrapped = self.make_counting_range()

        g1 = wrapped(3)
        g2 = wrapped(3)

        assert g1 is not g2

    def test_replay_does_not_advance_underlying_generator(self):
        """Replaying cached items should not re-run the underlying generator."""
        wrapped = self.make_counting_range()

        g1 = wrapped(5)
        assert next(g1) == 0
        assert next(g1) == 1
        assert self.created == 1
        assert self.yielded == 2

        # New call should replay 0,1 without advancing underlying generator
        g2 = wrapped(5)
        assert next(g2) == 0
        assert next(g2) == 1
        assert self.created == 1
        assert self.yielded == 2

    def test_shared_continuation_and_late_replay(self):
        """One iterator can extend the shared cache; others should replay newly cached items."""
        wrapped = self.make_counting_range()

        g1 = wrapped(5)
        assert [next(g1), next(g1)] == [0, 1]
        assert self.created == 1
        assert self.yielded == 2

        # Second iterator replays prefix and then advances the underlying generator
        g2 = wrapped(5)
        assert list(g2) == [0, 1, 2, 3, 4]
        assert self.created == 1
        assert self.yielded == 5

        # First iterator should now be able to finish purely from cache
        assert list(g1) == [2, 3, 4]
        assert self.created == 1
        assert self.yielded == 5

    def test_completion_is_cached(self):
        """Once a generator finishes, subsequent calls should only replay cached items."""
        wrapped = self.make_counting_range()

        assert list(wrapped(3)) == [0, 1, 2]
        assert self.created == 1
        assert self.yielded == 3

        # Call again; should not create/advance underlying generator
        assert list(wrapped(3)) == [0, 1, 2]
        assert self.created == 1
        assert self.yielded == 3


class TestSignatureKeying(CacheGeneratorTests):
    """Tests for cache key behavior (args/kwargs signature)."""

    def test_different_args_have_independent_caches(self):
        """Different positional args should map to different cache entries."""
        wrapped = self.make_counting_range()

        assert list(wrapped(2)) == [0, 1]
        assert list(wrapped(3)) == [0, 1, 2]

        # Each distinct signature should create its own underlying generator
        assert self.created == 2
        assert self.yielded == 5

    def test_sorted_kwargs_order_does_not_matter(self):
        """Kwarg order should not affect cache identity."""

        created = 0
        yielded = 0

        def _gen(*, a: int, b: int):
            nonlocal created, yielded
            created += 1
            yielded += 1
            yield a + b

        wrapped = cache_generator(_gen)

        assert list(wrapped(a=1, b=2)) == [3]
        # Same signature, different kwarg order
        assert list(wrapped(b=2, a=1)) == [3]

        assert created == 1
        assert yielded == 1

    def test_positional_vs_keyword_is_distinct_signature(self):
        """Like functools.cache, different call signatures should cache separately."""

        created = 0

        def _gen(a: int, *, b: int):
            nonlocal created
            created += 1
            yield (a, b)

        wrapped = cache_generator(_gen)

        assert list(wrapped(1, b=2)) == [(1, 2)]
        # Different signature (args differs): keyword-form for 'a'
        assert list(wrapped(a=1, b=2)) == [(1, 2)]

        # These are different signatures (args differs), so they should not share cache.
        assert created == 2


class TestExceptionCaching(CacheGeneratorTests):
    """Tests for caching terminal exceptions."""

    def test_exception_is_cached_and_reraised_on_future_calls(self):
        """If the underlying generator raises, the exception should be cached and re-raised."""
        wrapped = self.make_error_range()

        g1 = wrapped(10, error_at=2)
        assert next(g1) == 0
        assert next(g1) == 1

        with pytest.raises(RuntimeError) as e1:
            next(g1)

        assert str(e1.value) == "error occurred"
        assert self.created == 1
        assert self.yielded == 2

        # Future callers should replay cached prefix and then raise the *same* exception instance
        g2 = wrapped(10, error_at=2)
        assert next(g2) == 0
        assert next(g2) == 1

        with pytest.raises(RuntimeError) as e2:
            next(g2)

        assert e2.value is e1.value
        assert self.created == 1
        assert self.yielded == 2

    def test_exception_after_partial_consumption_by_other_caller(self):
        """A later caller can hit the exception; earlier callers should see it when they catch up."""
        wrapped = self.make_error_range()

        g1 = wrapped(10, error_at=3)
        assert next(g1) == 0  # cache now has [0]

        g2 = wrapped(10, error_at=3)
        # Advance g2 a bit (without running it to completion / exception yet)
        assert next(g2) == 0
        assert next(g2) == 1
        assert self.created == 1
        assert self.yielded == 2

        # Continue g2 to the point of failure
        assert next(g2) == 2
        assert self.yielded == 3

        with pytest.raises(RuntimeError) as e2:
            next(g2)

        assert str(e2.value) == "error occurred"

        # Now g1 should be able to replay newly cached items, then see the same exception
        assert next(g1) == 1
        assert next(g1) == 2

        with pytest.raises(RuntimeError) as e1:
            next(g1)

        assert e1.value is e2.value
        assert self.created == 1
        assert self.yielded == 3
