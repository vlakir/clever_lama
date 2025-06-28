"""Main application module for CleverLama."""

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import httpx
import uvicorn
from config import settings
from constants import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_SERVER_HEADER,
    HEALTH_PROBE_DELAY,
    RESPONSE_PREFIX,
)
from endpoints import ollama_router
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from logger import logger
from models import (
    OllamaHealthResponse,
)
from service import OpenAIService, client_holder

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])

service = OpenAIService()


async def startup_event() -> None:
    """Событие запуска приложения."""
    try:
        client_holder.client = httpx.AsyncClient(
            base_url=settings.api_base_url,
            headers={'Authorization': f'Bearer {settings.api_key}'},
            timeout=settings.request_timeout,
        )
        logger.info(f'✅ HTTP клиент инициализирован для {settings.api_base_url}')

        await asyncio.sleep(HEALTH_PROBE_DELAY)
        await service.health_check_external_api()

    except Exception:
        logger.exception('❌ Ошибка при запуске приложения')
        await shutdown_event()
        raise


async def shutdown_event() -> None:
    """Событие остановки приложения."""
    if client_holder.client:
        await client_holder.client.aclose()
        logger.info('✅ HTTP клиент закрыт')


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Управление жизненным циклом приложения."""
    # Startup
    logger.info('🚀 Запуск CleverLama API моста')
    await startup_event()

    yield

    # Shutdown
    logger.info('🛑 Остановка CleverLama API моста')
    await shutdown_event()


# Создание FastAPI приложения
app = FastAPI(
    title='CleverLama',
    version='1.0.0',
    description='Мост между Ollama API и OpenAI-совместимыми провайдерами',
    docs_url=None,  # Отключаем swagger - как y настоящего Ollama
    redoc_url=None,  # Отключаем redoc
    lifespan=lifespan,
)

# CORS middleware
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
    """Обработчик ошибок валидации."""
    logger.error(f'❌ Ошибка валидации для {request.url.path}')
    logger.error(f'❌ Детали ошибок: {exc.errors()}')

    return JSONResponse(
        status_code=422,
        content={
            'detail': f' {RESPONSE_PREFIX} Ошибка валидации данных',
            'errors': exc.errors(),
        },
    )


@app.middleware('http')
async def add_ollama_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Добавляет заголовки для совместимости с Ollama."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Добавляем заголовки как y настоящего Ollama
    response.headers['server'] = DEFAULT_SERVER_HEADER
    response.headers['content-type'] = DEFAULT_CONTENT_TYPE
    response.headers['x-process-time'] = str(process_time)

    return response


app.include_router(ollama_router)


# Этот эндпойнт проверяет AI Assistant при первом соединении
@app.get('/')
async def health_check() -> OllamaHealthResponse:
    """Health check endpoint."""
    return OllamaHealthResponse()


if __name__ == '__main__':
    python_version = sys.version.split()[0]
    logger.info(f'🐍 Python версия: {python_version}')
    logger.info(f'⚙️ Конфигурация: {settings.model_dump_json(indent=2)}')

    uvicorn.run(
        'main:app',
        host=settings.host,
        port=settings.port,
        log_level='info',
        access_log=True,
    )
