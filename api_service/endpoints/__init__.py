"""
CivilityAI: API Routers Export.
"""

from api_service.endpoints.analytics_routes import analytics_router
from api_service.endpoints.batch_routes import batch_router
from api_service.endpoints.health_routes import health_router
from api_service.endpoints.review_routes import review_router
from api_service.endpoints.safety_routes import safety_router

__all__ = [
    "analytics_router",
    "batch_router",
    "health_router",
    "review_router",
    "safety_router",
]
