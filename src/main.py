"""Main application module for CleverLama."""

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from config import settings
from constants import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_SERVER_HEADER,
    HEALTH_PROBE_DELAY,
    RESPONSE_PREFIX,
)
from logger import logger
from ports.api.ollama.endpoints import ollama_router, root_router
from ports.spi.openai.gateway import OpenAIGateway, client_holder
from services.proxy import OpenAIService

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])

service = OpenAIService()
gateway = OpenAIGateway()


def custom_key_builder(  # noqa: PLR0913
    func: Callable[..., Any],
    namespace: str = '',  # noqa: ARG001
    request: Request | None = None,  # noqa: ARG001
    response: Response | None = None,  # noqa: ARG001
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> str:
    if kwargs is None:
        kwargs = {}
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ['service', 'request', 'response', 'args', 'kwargs']
    }
    return f'{func.__module__}:{func.__name__}:{args}:{filtered_kwargs}'


async def startup_event() -> None:
    """Application startup event."""
    try:
        client_holder.client = httpx.AsyncClient(
            base_url=settings.api_base_url,
            headers={'Authorization': f'Bearer {settings.api_key}'},
            timeout=settings.request_timeout,
        )
        logger.info(f'✅ HTTP client initialized for {settings.api_base_url}')

        FastAPICache.init(
            InMemoryBackend(), prefix='fastapi-cache', key_builder=custom_key_builder
        )
        logger.info('✅ Model cache initialized')

        await asyncio.sleep(HEALTH_PROBE_DELAY)

        await gateway.health_check_external_api()

    except Exception:
        logger.exception('❌ Error during application startup')
        await shutdown_event()
        raise


async def shutdown_event() -> None:
    """Application shutdown event."""
    if client_holder.client:
        await client_holder.client.aclose()
        logger.info('✅ HTTP client closed')


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle management."""
    # Startup
    logger.info('🚀 Starting CleverLama API bridge')
    await startup_event()

    yield

    # Shutdown
    logger.info('🛑 Stopping CleverLama API bridge')
    await shutdown_event()


app = FastAPI(
    title='CleverLama',
    version='1.0.0',
    description='Bridge between Ollama API and OpenAI-compatible providers',
    docs_url=None,  # Disable swagger - like real Ollama
    redoc_url=None,  # Disable redoc
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors."""
    logger.error(f'❌ Validation error for {request.url.path}')
    logger.error(f'❌ Error details: {exc.errors()}')

    return JSONResponse(
        status_code=422,
        content={
            'detail': f' {RESPONSE_PREFIX} Data validation error',
            'errors': exc.errors(),
        },
    )


@app.middleware('http')
async def add_ollama_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Add headers for Ollama compatibility."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Add headers like real Ollama
    response.headers['server'] = DEFAULT_SERVER_HEADER
    response.headers['content-type'] = DEFAULT_CONTENT_TYPE
    response.headers['x-process-time'] = str(process_time)

    return response


app.include_router(ollama_router)
app.include_router(root_router)


if __name__ == '__main__':
    python_version = sys.version.split()[0]
    logger.info(f'🐍 Python version: {python_version}')
    logger.info(f'⚙️ Configuration: {settings.model_dump_json(indent=2)}')

    uvicorn.run(
        'main:app',
        host=settings.host,
        port=settings.port,
        log_level='info',
        access_log=True,
    )
