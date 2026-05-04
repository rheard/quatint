# TODO: Move this to a proper package? Maybe join with splitcache into an infcache library for infinite caches?

from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import Any, Callable, Generator, Hashable, TypeVar, cast

T = TypeVar("T")

# Sentinel to separate kwargs from args in the cache key
_KW_MARKER = object()


class _GenCacheEntry:
    """mypyc-friendly cache entry container (avoid @dataclass)"""

    __slots__ = ("lock", "items", "gen", "done", "exc")

    def __init__(self, gen: Generator[Any, None, None]):
        self.lock = RLock()
        self.items: list[Any] = []
        self.gen: Generator[Any, None, None] | None = gen
        self.done: bool = False
        self.exc: BaseException | None = None


def _make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    """Cache key similar to functools.cache: args + sorted kwargs."""
    if not kwargs:
        return args
    # Sorting makes the key deterministic
    return *args, _KW_MARKER, *tuple(sorted(kwargs.items()))


def cache_generator(
    fn: Callable[..., Generator[T, None, None]],
) -> Callable[..., Generator[T, None, None]]:
    """
    Cache a generator function like functools.cache caches a normal function.

    For each unique call signature, we store:
      - the list of items yielded so far
      - the underlying generator (if still running)
      - completion status and terminal exception (if any)

    Each invocation returns a *fresh generator* that:
      - replays cached items
      - then continues consuming the shared underlying generator, caching new yields
      - if/when the underlying generator terminates with an exception, it is cached and
        re-raised after replay/continuation on all future calls.

    Returns:
        Callable: The wrapped method.
    """
    cache: dict[Hashable, _GenCacheEntry] = {}
    cache_lock = RLock()  # guards cache dict insertion/lookup

    @wraps(fn)
    def wrapper(*args: tuple[Any], **kwargs: dict[str, Any]) -> Generator[T, None, None]:
        key = _make_key(args, kwargs)

        # Get/create entry
        with cache_lock:
            entry = cache.get(key)
            if entry is None:
                entry = _GenCacheEntry(cast("Generator[Any, None, None]", fn(*args, **kwargs)))
                cache[key] = entry

        def _iter() -> Generator[T, None, None]:
            i = 0
            while True:
                # Step 1: replay anything already cached (no lock needed for reading length,
                # but we need lock to ensure i < len(items) check and item access are consistent)
                with entry.lock:
                    n = len(entry.items)
                    if i < n:
                        item = entry.items[i]
                        i += 1
                        # Yield outside lock
                    else:
                        item = None

                if item is not None:
                    yield cast("T", item)
                    continue

                # Step 2: no more cached prefix; either finished or we must advance shared generator
                with entry.lock:
                    if entry.done:
                        if entry.exc is not None:
                            raise entry.exc
                        return

                    gen = entry.gen
                    if gen is None:
                        # Should not happen, but keep behavior sane
                        entry.done = True
                        return

                    # Advance the shared generator by exactly one item under lock so only one caller
                    # performs the step and caches the result.
                    try:
                        nxt = next(gen)
                    except StopIteration:
                        entry.gen = None
                        entry.done = True
                        entry.exc = None
                        return
                    except BaseException as e:
                        entry.gen = None
                        entry.done = True
                        entry.exc = e
                        raise
                    else:
                        entry.items.append(nxt)
                        i += 1  # we are about to yield this new cached item

                yield cast("T", nxt)

        return _iter()

    return wrapper
