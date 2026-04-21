"""FastAPI application entry point.

Registers all routers and configures structured logging.
No endpoint logic lives here — see routers/.
"""

import logging
import logging.config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import get_settings
from app.routers import health, priority, query

_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    settings = get_settings()

    logging.config.dictConfig(_LOG_CONFIG)
    logging.getLogger().setLevel(settings.log_level.upper())

    app = FastAPI(
        title="Decision Intelligence Assistant",
        description="RAG + ML + LLM four-way comparison over customer support tickets",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(priority.router)

    return app


app = create_app()
