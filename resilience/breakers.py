import pybreaker
import structlog

log = structlog.get_logger()


pageindex_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=30,
    name="pageindex",
)

gptcache_breaker = pybreaker.CircuitBreaker(
    fail_max=10,
    reset_timeout=15,
    name="gptcache",
)
