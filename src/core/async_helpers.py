"""
Structured Async Helpers
Red Team Amendment: Standardized async patterns with timeout and cancellation
"""
import asyncio
import logging
from typing import Any, Awaitable, List, TypeVar
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
T = TypeVar('T')

async def timeout(coro: Awaitable[T], seconds: float) -> T:
    """Wrap asyncio.wait_for with consistent logging"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds}s")
        raise TimeoutError(f"Operation exceeded {seconds}s timeout")

async def bounded(pool_sem: asyncio.Semaphore, coro: Awaitable[T]) -> T:
    """Ensure semaphore wrapping for bounded concurrency"""
    async with pool_sem:
        return await coro

async def cancel_on_shutdown(tasks: List[asyncio.Task]) -> None:
    """Coordinated cancellation of task list"""
    logger.info(f"Cancelling {len(tasks)} tasks")
    
    # Cancel all tasks
    for task in tasks:
        if not task.done():
            task.cancel()
    
    # Wait for graceful cancellation
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.info("All tasks cancelled successfully")

@asynccontextmanager
async def timeout_context(seconds: float):
    """Context manager for timeout operations"""
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.warning(f"Context operation timed out after {seconds}s")
        raise TimeoutError(f"Context exceeded {seconds}s timeout")

class BoundedConcurrencyManager:
    """Manager for bounded concurrent operations"""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks: List[asyncio.Task] = []
    
    async def execute(self, coro: Awaitable[T]) -> T:
        """Execute with bounded concurrency"""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.active_tasks.append(task)
            try:
                result = await task
                return result
            finally:
                self.active_tasks.remove(task)
    
    async def shutdown(self):
        """Cancel all active tasks"""
        await cancel_on_shutdown(self.active_tasks)
        self.active_tasks.clear()

# Performance monitoring helper
async def monitor_slow_calls(coro: Awaitable[T], operation_name: str, p95_threshold: float = 0.4) -> T:
    """Monitor and log slow operations"""
    import time
    start_time = time.time()
    
    try:
        result = await coro
        duration = time.time() - start_time
        
        if duration > p95_threshold:
            logger.warning(f"Slow operation: {operation_name} took {duration:.2f}s (> {p95_threshold}s)")
        
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed operation: {operation_name} failed after {duration:.2f}s: {e}")
        raise