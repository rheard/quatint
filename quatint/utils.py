from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any, Callable, Generator, Generic, Hashable, Optional, TypeVar, cast

T = TypeVar("T")


@dataclass
class _GenCacheEntry(Generic[T]):
    lock: RLock
    items: list[T]
    gen: Optional[Generator[T, None, None]]
    done: bool
    exc: Optional[BaseException]


def _make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    """Cache key similar to functools.cache: args + sorted kwargs."""
    if not kwargs:
        return args
    # Sorting makes the key deterministic
    return *args, object(), *tuple(sorted(kwargs.items()))


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
    cache: dict[Hashable, _GenCacheEntry[T]] = {}
    cache_lock = RLock()  # guards cache dict insertion/lookup

    @wraps(fn)
    def wrapper(*args: tuple[Any], **kwargs: dict[str, Any]) -> Generator[T, None, None]:
        key = _make_key(args, kwargs)

        # Get/create entry
        with cache_lock:
            entry = cache.get(key)
            if entry is None:
                entry = _GenCacheEntry(
                    lock=RLock(),
                    items=[],
                    gen=fn(*args, **kwargs),
                    done=False,
                    exc=None,
                )
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

                yield nxt

        return _iter()

    return wrapper
