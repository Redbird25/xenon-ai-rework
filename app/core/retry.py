"""
Retry mechanisms and circuit breakers for external service calls
"""
import asyncio
import time
from typing import TypeVar, Callable, Optional, Any, Dict
from functools import wraps
from enum import Enum
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import structlog

import logging as py_logging
from app.core.exceptions import RateLimitError, AIIngestException
from app.config import settings

logger = structlog.get_logger(__name__)
py_logger = py_logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker"""
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise AIIngestException(
                        f"Circuit breaker is OPEN for {func.__name__}",
                        {"state": self.state.value, "failures": self.failure_count}
                    )
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise AIIngestException(
                        f"Circuit breaker is OPEN for {func.__name__}",
                        {"state": self.state.value, "failures": self.failure_count}
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker opened",
                failures=self.failure_count,
                threshold=self.failure_threshold
            )


# Retry strategies for different scenarios
def get_retry_decorator(
    max_attempts: int = 3,
    wait_strategy: str = "exponential",
    exceptions: tuple = (Exception,)
):
    """Get configured retry decorator"""
    
    if wait_strategy == "exponential":
        wait = wait_exponential(multiplier=1, min=4, max=60)
    elif wait_strategy == "random_exponential":
        wait = wait_random_exponential(multiplier=1, max=60)
    else:
        wait = wait_exponential(multiplier=1, min=4, max=60)
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait,
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(py_logger, py_logging.WARNING),
        after=after_log(py_logger, py_logging.INFO)
    )


# Specific retry decorators for different services
llm_retry = get_retry_decorator(
    max_attempts=settings.llm_max_retries,
    wait_strategy="exponential",
    exceptions=(Exception,)
)

embedding_retry = get_retry_decorator(
    max_attempts=3,
    wait_strategy="exponential",
    exceptions=(Exception,)
)

http_retry = get_retry_decorator(
    max_attempts=3,
    wait_strategy="random_exponential",
    exceptions=(Exception,)
)


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_second: Optional[int] = None):
        self.calls_per_minute = calls_per_minute
        self.calls_per_second = calls_per_second or calls_per_minute // 60
        self.minute_calls = []
        self.second_calls = []
        
    async def acquire(self):
        """Acquire permission to make a call"""
        now = time.time()
        
        # Clean old calls
        self.minute_calls = [t for t in self.minute_calls if now - t < 60]
        self.second_calls = [t for t in self.second_calls if now - t < 1]
        
        # Check per-second limit
        if len(self.second_calls) >= self.calls_per_second:
            sleep_time = 1 - (now - self.second_calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        # Check per-minute limit
        if len(self.minute_calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.minute_calls[0])
            if sleep_time > 0:
                logger.warning(
                    "Rate limit reached, sleeping",
                    sleep_seconds=sleep_time,
                    calls_per_minute=self.calls_per_minute
                )
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        # Record call
        self.minute_calls.append(now)
        self.second_calls.append(now)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator for rate limiting"""
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await self.acquire()
            return await func(*args, **kwargs)
        
        return wrapper


class RetryWithBackoff:
    """Advanced retry with backoff and jitter"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry with backoff"""
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_attempts - 1:
                        logger.error(
                            "Max retry attempts reached",
                            function=func.__name__,
                            attempts=self.max_attempts,
                            error=str(e)
                        )
                        raise
                    
                    # Calculate delay
                    delay = min(
                        self.initial_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    if self.jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        "Retrying after error",
                        function=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_attempts - 1:
                        logger.error(
                            "Max retry attempts reached",
                            function=func.__name__,
                            attempts=self.max_attempts,
                            error=str(e)
                        )
                        raise
                    
                    # Calculate delay
                    delay = min(
                        self.initial_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    if self.jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        "Retrying after error",
                        function=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper


# Global instances
llm_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
embedding_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
llm_rate_limiter = RateLimiter(calls_per_minute=settings.rate_limit_rpm)
embedding_rate_limiter = RateLimiter(calls_per_minute=100)
