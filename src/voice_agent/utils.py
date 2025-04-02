import asyncio
from collections.abc import AsyncIterator
from typing import Any, TypeVar

T = TypeVar("T")


async def amerge(**iterators: AsyncIterator[Any]) -> AsyncIterator[tuple[str, Any]]:
    """Merge multiple async iterators into a single async iterator.

    Each item yielded is a tuple of (key, value) where key is the name of the
    iterator that produced the value.
    """
    tasks = {
        key: asyncio.create_task(anext(iterator)) for key, iterator in iterators.items()
    }
    try:
        while tasks:
            done, _ = await asyncio.wait(
                tasks.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                # Find which key this task belongs to
                key = next(k for k, t in tasks.items() if t == task)
                try:
                    result = task.result()
                    yield key, result
                    # Create new task for this iterator
                    tasks[key] = asyncio.create_task(anext(iterators[key]))
                except StopAsyncIteration:
                    # This iterator is done, remove it from tasks
                    del tasks[key]
                    if not tasks:  # All streams are done
                        return
    finally:
        # Cancel any remaining tasks
        for task in tasks.values():
            if not task.done():
                task.cancel()
        # Wait for all tasks to complete/cancel
        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)
