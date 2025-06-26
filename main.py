#!/usr/bin/env python3
"""Entry point for CleverLama application."""

if __name__ == '__main__':
    import uvicorn

    from src.clever_lama.config import settings
    from src.clever_lama.main import app

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
