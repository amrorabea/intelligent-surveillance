# Export all routers for easy importing
from .jobs import jobs_router
from .queries import queries_router
from .frames import frames_router
from .analytics import analytics_router
from .health import health_router

__all__ = ["jobs_router", "queries_router", "frames_router", "analytics_router", "health_router"]
