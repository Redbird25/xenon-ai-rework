"""
Structured logging and monitoring for AI Ingest Service
"""
import sys
import time
import json
from functools import wraps
from typing import Any, Dict, Optional, Callable
from contextvars import ContextVar
import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder
from prometheus_client import Counter, Histogram, Gauge
import logging

from app.config import settings

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
job_id_var: ContextVar[Optional[str]] = ContextVar("job_id", default=None)

# Prometheus metrics
ingest_requests = Counter("ingest_requests_total", "Total ingest requests", ["status"])
ingest_duration = Histogram("ingest_duration_seconds", "Ingest processing duration")
chunk_count = Counter("chunks_created_total", "Total chunks created")
embedding_requests = Counter("embedding_requests_total", "Total embedding requests", ["model", "status"])
embedding_duration = Histogram("embedding_duration_seconds", "Embedding generation duration", ["model"])
llm_requests = Counter("llm_requests_total", "Total LLM requests", ["model", "operation", "status"])
llm_duration = Histogram("llm_duration_seconds", "LLM request duration", ["model", "operation"])
active_jobs = Gauge("active_ingest_jobs", "Number of active ingest jobs")


def add_request_context(logger, method_name, event_dict):
    """Add request context to log events"""
    request_id = request_id_var.get()
    job_id = job_id_var.get()
    
    if request_id:
        event_dict["request_id"] = request_id
    if job_id:
        event_dict["job_id"] = job_id
    
    # Add service metadata
    event_dict["service"] = "ai-ingest"
    event_dict["environment"] = settings.environment.value
    
    return event_dict


def setup_logging():
    """Configure structured logging for the application"""
    
    # Configure Python logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper())
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_request_context,
        CallsiteParameterAdder(
            parameters=[CallsiteParameter.FILENAME, CallsiteParameter.LINENO]
        ),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Use JSON in production
    if settings.log_format == "json" or settings.is_production:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log and measure function execution time"""
    logger = get_logger(func.__module__)
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        logger.info("function_start", function=function_name)
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info("function_success",
                       function=function_name,
                       duration_seconds=duration)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error("function_error",
                        function=function_name,
                        duration_seconds=duration,
                        error=str(e),
                        error_type=type(e).__name__)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        logger.info("function_start", function=function_name)
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info("function_success",
                       function=function_name,
                       duration_seconds=duration)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error("function_error",
                        function=function_name,
                        duration_seconds=duration,
                        error=str(e),
                        error_type=type(e).__name__)
            raise
    
    # Return appropriate wrapper
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class RequestLogger:
    """Context manager for request logging"""
    
    def __init__(self, request_id: str, operation: str):
        self.request_id = request_id
        self.operation = operation
        self.logger = get_logger(__name__)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        request_id_var.set(self.request_id)
        self.logger.info("request_start", operation=self.operation)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info("request_complete",
                           operation=self.operation,
                           duration_seconds=duration)
        else:
            self.logger.error("request_failed",
                            operation=self.operation,
                            duration_seconds=duration,
                            error=str(exc_val),
                            error_type=exc_type.__name__)
        
        request_id_var.set(None)


class MetricsLogger:
    """Helper class for logging with metrics"""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
        
    def log_ingest_start(self, job_id: str, resource_count: int):
        """Log ingest job start"""
        job_id_var.set(job_id)
        active_jobs.inc()
        self.logger.info("ingest_job_started",
                        job_id=job_id,
                        resource_count=resource_count)
        
    def log_ingest_complete(self, job_id: str, duration: float, chunk_count: int):
        """Log ingest job completion"""
        active_jobs.dec()
        ingest_requests.labels(status="success").inc()
        ingest_duration.observe(duration)
        self.logger.info("ingest_job_completed",
                        job_id=job_id,
                        duration_seconds=duration,
                        chunks_created=chunk_count)
        
    def log_ingest_error(self, job_id: str, error: str):
        """Log ingest job error"""
        active_jobs.dec()
        ingest_requests.labels(status="error").inc()
        self.logger.error("ingest_job_failed",
                         job_id=job_id,
                         error=error)
        
    def log_embedding_request(self, model: str, text_count: int):
        """Log embedding request"""
        self.logger.info("embedding_request",
                        model=model,
                        text_count=text_count)
        
    def log_embedding_complete(self, model: str, duration: float, success: bool = True):
        """Log embedding completion"""
        status = "success" if success else "error"
        embedding_requests.labels(model=model, status=status).inc()
        embedding_duration.labels(model=model).observe(duration)
        
        if success:
            self.logger.info("embedding_complete",
                           model=model,
                           duration_seconds=duration)
        else:
            self.logger.error("embedding_failed",
                            model=model,
                            duration_seconds=duration)
            
    def log_llm_request(self, model: str, operation: str, prompt_length: int):
        """Log LLM request"""
        self.logger.info("llm_request",
                        model=model,
                        operation=operation,
                        prompt_length=prompt_length)
        
    def log_llm_complete(self, model: str, operation: str, duration: float, 
                        tokens_used: int = 0, success: bool = True):
        """Log LLM completion"""
        status = "success" if success else "error"
        llm_requests.labels(model=model, operation=operation, status=status).inc()
        llm_duration.labels(model=model, operation=operation).observe(duration)
        
        if success:
            self.logger.info("llm_complete",
                           model=model,
                           operation=operation,
                           duration_seconds=duration,
                           tokens_used=tokens_used)
        else:
            self.logger.error("llm_failed",
                            model=model,
                            operation=operation,
                            duration_seconds=duration)
            
    def log_chunk_creation(self, count: int, strategy: str, avg_size: float):
        """Log chunk creation"""
        chunk_count.inc(count)
        self.logger.info("chunks_created",
                        count=count,
                        strategy=strategy,
                        avg_size=avg_size)


# Initialize logging on module import
setup_logging()

# Create default logger
logger = get_logger(__name__)
metrics_logger = MetricsLogger(logger)
