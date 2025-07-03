import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi_cache.decorator import cache

from clever_lama.constants import (
    API_VERSION,
    CACHE_EXPIRATION_S,
    HTTP_BAD_GATEWAY_ERROR,
)
from clever_lama.logger import logger
from clever_lama.models.ollama import OllamaMessage, OllamaModelDetails
from clever_lama.ports.api.ollama.depends import OpenAIServiceDep
from clever_lama.ports.api.ollama.schemas import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaHealthResponse,
    OllamaModelsResponse,
    OllamaPullRequest,
    OllamaPullResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaVersionResponse,
)

root_router = APIRouter(
    prefix='',
    tags=['root'],
)


ollama_router = APIRouter(
    prefix='/api',
    tags=['ollama'],
)


# Этот эндпойнт проверяет AI Assistant при первом соединении
@root_router.get('/')
async def health_check() -> OllamaHealthResponse:
    """Health check endpoint."""
    return OllamaHealthResponse()


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
                status_code=HTTP_BAD_GATEWAY_ERROR, detail=str(e)
            ) from e

    return wrapper  # type: ignore[return-value]


@ollama_router.get('/version')
@error_handler
async def get_version() -> OllamaVersionResponse:
    """Возвращает версию API."""
    return OllamaVersionResponse(version=API_VERSION)


@ollama_router.get('/tags')
@error_handler
@cache(expire=CACHE_EXPIRATION_S)
async def tags(service: OpenAIServiceDep) -> OllamaModelsResponse | None:
    """Возвращает список доступных моделей."""
    logger.info('Получен запрос к /api/tags')

    models = await service.get_models()

    logger.info(f'Получено {len(models) if models else 0} моделей')

    return OllamaModelsResponse(models=models)


@ollama_router.post('/show')
@error_handler
async def show(request: OllamaShowRequest) -> OllamaShowResponse:
    """Возвращает информацию о модели."""
    return OllamaShowResponse(
        license='MIT',
        modelfile=f'FROM {request.name}',
        parameters='temperature 0.7\ntop_p 0.9',
        template='{{ .Prompt }}',
        details=OllamaModelDetails.model_validate(
            {
                'parent_model': '',
                'format': 'gguf',
                'family': 'llama',
                'families': ['llama'],
                'parameter_size': '7B',
                'quantization_level': 'Q4_0',
                'num_parameters': 7_000_000_000,
            }
        ),
    )


@ollama_router.post('/pull')
@error_handler
async def pull_model(request: OllamaPullRequest) -> OllamaPullResponse:
    """Имитирует загрузку модели."""
    return OllamaPullResponse(
        status=f'pulling {request.name}', total=1000000000, completed=1000000000
    )


@ollama_router.post('/generate')
@error_handler
async def generate(
    request: OllamaGenerateRequest, service: OpenAIServiceDep
) -> OllamaGenerateResponse:
    """Генерирует текст используя внешний API."""
    messages = [{'role': 'user', 'content': request.prompt}]

    generated_text = await service.call_api(
        messages, request.model, stream=request.stream
    )

    return OllamaGenerateResponse(
        model=request.model,
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        response=generated_text,
        done=True,
        context=[],
        total_duration=1000000000,
        load_duration=100000000,
        prompt_eval_count=len(request.prompt.split()),
        prompt_eval_duration=50000000,
        eval_count=len(generated_text.split()),
        eval_duration=900000000,
    )


@ollama_router.post('/chat', response_model=None)
@error_handler
async def chat(
    request: OllamaChatRequest, service: OpenAIServiceDep
) -> OllamaChatResponse | StreamingResponse:
    """Чат с моделью используя внешний API."""
    # Добавляем детальное логирование
    logger.info(
        f'Получен запрос к /api/chat: model={request.model}, '
        f'messages_count={len(request.messages)}, stream={request.stream}'
    )
    logger.debug(f'Полный запрос: {request.model_dump()}')

    try:
        openai_messages = [
            {'role': msg.role, 'content': msg.content} for msg in request.messages
        ]

        logger.debug(f'Трансформированные сообщения: {openai_messages}')

        # Если запрошена потоковая передача
        if request.stream:
            logger.info('Обработка потокового запроса')

            return StreamingResponse(
                service.get_stream(messages=openai_messages, model=request.model),
                media_type='application/x-ndjson',
                headers={'Cache-Control': 'no-cache'},
            )

        # Обычный ответ без потоковой передачи
        generated_text = await service.call_api(
            openai_messages, request.model, stream=False
        )

        logger.info(f'Получен ответ длиной {len(generated_text)} символов')

        return OllamaChatResponse(
            model=request.model,
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            message=OllamaMessage(role='assistant', content=generated_text),
            done=True,
        )
    except Exception as e:
        logger.error(f'Ошибка в chat_with_model: {e!s}', exc_info=True)

        raise HTTPException(status_code=HTTP_BAD_GATEWAY_ERROR, detail=str(e)) from e
