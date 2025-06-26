"""Main application module for CleverLama API bridge."""

import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from types import MappingProxyType
from typing import Any, TypeVar

import httpx
import uvicorn
from config import settings
from constants import (
    API_VERSION,
    DEFAULT_CONTENT_TYPE,
    DEFAULT_MODEL,
    DEFAULT_SERVER_HEADER,
    HEALTH_PROBE_DELAY,
    HTTP_OK,
    HTTP_SERVER_ERROR,
    SERVER_ERROR_MESSAGE,
)
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from models import (
    ErrorResponse,
    HealthResponse,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaModel,
    OllamaModelsResponse,
    OllamaPullRequest,
    OllamaPullResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    VersionResponse,
)

# Константы
CACHE_MODELS_KEY = 'models_cache'
CACHE_MODELS_TIMESTAMP_KEY = 'models_cache_timestamp'

# Типы
F = TypeVar('F', bound=Callable[..., Awaitable[Any]])


class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для логов."""

    # ANSI escape коды для цветов
    COLORS = MappingProxyType(
        {
            'DEBUG': '\033[36m',  # Cyan
            'INFO': '\033[0m',  # Default/White (вместо красного)
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',  # Red (только ERROR будет красным)
            'CRITICAL': '\033[35m',  # Magenta
            'RESET': '\033[0m',  # Reset
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Получаем исходное сообщение
        log_message = super().format(record)

        # Добавляем цвет в зависимости от уровня
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        return f'{color}{log_message}{reset}'


# Настройка логирования с цветным форматтером
def setup_logging() -> None:
    """Настраивает логирование с цветным выводом."""
    # Создаем форматтер
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Получаем root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Удаляем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Создаем новый обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Добавляем обработчик к root logger
    root_logger.addHandler(console_handler)


# Инициализируем логирование
setup_logging()
logger = logging.getLogger(__name__)

# Кэш для моделей
cache: dict[str, Any] = {}


# HTTP клиент holder
class HTTPClientHolder:
    """Контейнер для HTTP клиента, избегаем global."""

    def __init__(self) -> None:
        self.client: httpx.AsyncClient | None = None


client_holder = HTTPClientHolder()


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


async def startup_event() -> None:
    """Событие запуска приложения."""
    try:
        client_holder.client = httpx.AsyncClient(
            base_url=settings.api_base_url,
            headers={'Authorization': f'Bearer {settings.api_key}'},
            timeout=settings.request_timeout,
        )
        logger.info(f'✅ HTTP клиент инициализирован для {settings.api_base_url}')

        # Проводим health check внешнего API
        await asyncio.sleep(HEALTH_PROBE_DELAY)
        await health_check_external_api()

    except Exception:
        logger.exception('❌ Ошибка при запуске приложения')
        await shutdown_event()
        raise


async def shutdown_event() -> None:
    """Событие остановки приложения."""
    if client_holder.client:
        await client_holder.client.aclose()
        logger.info('✅ HTTP клиент закрыт')


def error_handler[F: Callable[..., Awaitable[Any]]](func: F) -> F:
    """Декоратор для обработки ошибок в endpoint'ах."""

    @wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Перебрасываем HTTPException как есть
            raise
        except Exception as e:
            logger.exception('Неожиданная ошибка в endpoint')
            raise HTTPException(
                status_code=HTTP_SERVER_ERROR,
                detail=ErrorResponse(
                    detail=f'{SERVER_ERROR_MESSAGE}: {e}',
                    error_code='INTERNAL_ERROR',
                ).model_dump(),
            ) from e

    return wrapper  # type: ignore[return-value]


async def health_check_external_api() -> None:
    """Проверяет доступность внешнего API."""
    if not client_holder.client:
        msg = 'HTTP клиент не инициализирован'
        raise RuntimeError(msg)

    try:
        response = await client_holder.client.get('/models')
        if response.status_code == HTTP_OK:
            logger.info('✅ Внешний API доступен')
        else:
            logger.warning(f'⚠️ Внешний API вернул статус {response.status_code}')
    except Exception:
        logger.exception('❌ Внешний API недоступен')


def is_cache_valid() -> bool:
    """Проверяет актуальность кэша моделей."""
    if CACHE_MODELS_TIMESTAMP_KEY not in cache:
        return False

    timestamp = cache[CACHE_MODELS_TIMESTAMP_KEY]
    if not isinstance(timestamp, (int, float)):
        return False

    current_time = time.time()
    cache_age_minutes = (current_time - timestamp) / 60
    return cache_age_minutes < settings.cache_duration_minutes


async def get_models_from_api() -> list[dict[str, Any]]:
    """Получает список моделей из внешнего API."""
    if not client_holder.client:
        msg = 'HTTP клиент не инициализирован'
        raise RuntimeError(msg)

    try:
        response = await client_holder.client.get('/models')
        response.raise_for_status()

        data = response.json()
        models = []

        if 'data' in data:
            for model in data['data']:
                model_id = model.get('id', DEFAULT_MODEL)
                models.append(
                    {
                        'id': model_id,
                        'object': 'model',
                        'created': str(int(time.time())),
                        'owned_by': 'aitunnel',
                    }
                )

    except Exception:
        logger.exception('Ошибка получения моделей из API')
        # Возвращаем модель по умолчанию
        models = [
            {
                'id': DEFAULT_MODEL,
                'object': 'model',
                'created': str(int(time.time())),
                'owned_by': 'aitunnel',
            }
        ]

    return models


async def get_cached_models() -> list[dict[str, Any]]:
    """Получает модели из кэша или загружает их."""
    if is_cache_valid() and CACHE_MODELS_KEY in cache:
        models = cache[CACHE_MODELS_KEY]
        if isinstance(models, list):
            return models

    # Загружаем модели из API
    models = await get_models_from_api()
    cache[CACHE_MODELS_KEY] = models
    cache[CACHE_MODELS_TIMESTAMP_KEY] = time.time()

    return models


def create_ollama_models_response(models: list[dict[str, Any]]) -> OllamaModelsResponse:
    """Создает ответ в формате Ollama из списка моделей."""
    ollama_models = []

    for model in models:
        model_id = model.get('id', DEFAULT_MODEL)
        created_time_str = model.get('created', str(int(time.time())))

        # Преобразуем created в int, если это строка
        try:
            created_time = (
                int(created_time_str)
                if isinstance(created_time_str, str)
                else created_time_str
            )
        except (ValueError, TypeError):
            created_time = int(time.time())

        # Создаем модель в формате Ollama
        ollama_model = OllamaModel(
            name=model_id,
            model=model_id,
            modified_at=time.strftime(
                '%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(float(created_time))
            ),
            size=1000000000,  # 1GB примерный размер
            digest=f'sha256:{"0" * 64}',  # Dummy digest
            details={
                'parent_model': '',
                'format': 'gguf',
                'family': 'llama',
                'families': ['llama'],
                'parameter_size': '7B',
                'quantization_level': 'Q4_0',
            },
        )
        ollama_models.append(ollama_model)

    return OllamaModelsResponse(models=ollama_models)


def transform_messages_for_openai(
    messages: list[OllamaMessage],
) -> list[dict[str, str]]:
    """Преобразует сообщения Ollama в формат OpenAI."""
    return [{'role': msg.role, 'content': msg.content} for msg in messages]


def extract_content_from_openai_response(response_data: dict[str, Any]) -> str:
    """Извлекает содержимое ответа из ответа OpenAI."""
    try:
        choices = response_data.get('choices', [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get('message', {})
                if isinstance(message, dict):
                    content = message.get('content', '')
                    if isinstance(content, str):
                        return content
    except (IndexError, KeyError, TypeError):
        pass

    return ''


async def call_openai_api(
    messages: list[dict[str, str]], model: str, *, stream: bool = False
) -> dict[str, Any]:
    """Вызывает OpenAI API для генерации ответа."""
    if not client_holder.client:
        msg = 'HTTP клиент не инициализирован'
        raise RuntimeError(msg)

    try:
        payload = {
            'model': model,
            'messages': messages,
            'stream': stream,
        }

        response = await client_holder.client.post('/chat/completions', json=payload)
        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as e:
        error_detail = f'OpenAI API error: {e.response.status_code}'
        try:
            error_data = e.response.json()
            if 'error' in error_data:
                error_detail = (
                    f'OpenAI API error: {error_data["error"].get("message", str(e))}'
                )
        except Exception:
            logger.exception('Ошибка при обработке ответа об ошибке API')

        raise HTTPException(
            status_code=e.response.status_code, detail=error_detail
        ) from e

    except Exception as e:
        logger.exception('Ошибка вызова OpenAI API')
        raise HTTPException(
            status_code=HTTP_SERVER_ERROR,
            detail=f'Ошибка подключения к OpenAI API: {e}',
        ) from e


# API эндпоинты
@app.get('/')
@error_handler
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@app.get('/api/version')
@error_handler
async def get_version() -> VersionResponse:
    """Возвращает версию API."""
    return VersionResponse(version=API_VERSION)


@app.get('/api/tags')
@error_handler
async def list_models() -> OllamaModelsResponse:
    """Возвращает список доступных моделей."""
    models = await get_cached_models()
    return create_ollama_models_response(models)


@app.post('/api/show')
@error_handler
async def show_model(request: OllamaShowRequest) -> OllamaShowResponse:
    """Возвращает информацию о модели."""
    return OllamaShowResponse(
        license='MIT',
        modelfile=f'FROM {request.name}',
        parameters='temperature 0.7\ntop_p 0.9',
        template='{{ .Prompt }}',
        details={
            'parent_model': '',
            'format': 'gguf',
            'family': 'llama',
            'families': ['llama'],
            'parameter_size': '7B',
            'quantization_level': 'Q4_0',
        },
    )


@app.post('/api/pull')
@error_handler
async def pull_model(request: OllamaPullRequest) -> OllamaPullResponse:
    """Имитирует загрузку модели."""
    return OllamaPullResponse(
        status=f'pulling {request.name}', total=1000000000, completed=1000000000
    )


@app.post('/api/generate')
@error_handler
async def generate_text(request: OllamaGenerateRequest) -> OllamaGenerateResponse:
    """Генерирует текст используя внешний API."""
    # Преобразуем prompt в формат сообщений для OpenAI
    messages = [{'role': 'user', 'content': request.prompt}]

    # Вызываем OpenAI API
    response_data = await call_openai_api(
        messages, request.model, stream=request.stream
    )

    # Извлекаем сгенерированный текст
    generated_text = extract_content_from_openai_response(response_data)

    # Формируем ответ в формате Ollama
    return OllamaGenerateResponse(
        model=request.model,
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        response=generated_text,
        done=True,
        context=[],
        total_duration=1000000000,  # 1 секунда в наносекундах
        load_duration=100000000,  # 0.1 секунды в наносекундах
        prompt_eval_count=len(request.prompt.split()),
        prompt_eval_duration=50000000,  # 0.05 секунды в наносекундах
        eval_count=len(generated_text.split()),
        eval_duration=900000000,  # 0.9 секунды в наносекундах
    )


@app.post('/api/chat')
@error_handler
async def chat_with_model(request: OllamaChatRequest) -> OllamaChatResponse:
    """Чат с моделью используя внешний API."""
    # Преобразуем сообщения в формат OpenAI
    openai_messages = transform_messages_for_openai(request.messages)

    # Вызываем OpenAI API
    response_data = await call_openai_api(
        openai_messages, request.model, stream=request.stream
    )

    # Извлекаем сгенерированный текст
    generated_text = extract_content_from_openai_response(response_data)

    # Формируем ответ в формате Ollama
    return OllamaChatResponse(
        model=request.model,
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        message=OllamaMessage(role='assistant', content=generated_text),
        done=True,
    )


if __name__ == '__main__':
    # Информация об Python
    python_version = sys.version.split()[0]
    logger.info(f'🐍 Python версия: {python_version}')
    logger.info(f'⚙️ Конфигурация: {settings.model_dump_json(indent=2)}')

    # Запуск сервера
    uvicorn.run(
        'main:app',
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
