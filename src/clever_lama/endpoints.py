import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Annotated, Any

from config import settings
from constants import (
    API_VERSION,
    HTTP_BAD_GATEWAY_ERROR,
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from logger import logger
from models import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaModelDetails,
    OllamaModelsResponse,
    OllamaPullRequest,
    OllamaPullResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaVersionResponse,
)
from service import OpenAIService

ollama_router = APIRouter(
    prefix='/api',
    tags=['ollama'],
)


OpenAIServiceDep = Annotated[OpenAIService, Depends(OpenAIService)]


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
async def list_models(service: OpenAIServiceDep) -> OllamaModelsResponse | None:
    """Возвращает список доступных моделей."""
    logger.info('Получен запрос к /api/tags')

    models = await service.get_models_from_api()
    logger.info(f'Получено {len(models) if models else 0} моделей')

    return service.create_ollama_models_response(models)


@ollama_router.post('/show')
@error_handler
async def show_model(request: OllamaShowRequest) -> OllamaShowResponse:
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
async def generate_text(
    request: OllamaGenerateRequest, service: OpenAIServiceDep
) -> OllamaGenerateResponse:
    """Генерирует текст используя внешний API."""
    messages = [{'role': 'user', 'content': request.prompt}]

    response_data = await service.call_openai_api(
        messages, request.model, stream=request.stream
    )

    generated_text = service.extract_content_from_openai_response(response_data)

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
async def chat_with_model(
    request: OllamaChatRequest, service: OpenAIServiceDep
) -> OllamaChatResponse | StreamingResponse | None:
    """Чат с моделью используя внешний API."""
    # Добавляем детальное логирование
    logger.info(
        f'Получен запрос к /api/chat: model={request.model}, '
        f'messages_count={len(request.messages)}, stream={request.stream}'
    )
    logger.debug(f'Полный запрос: {request.model_dump()}')

    try:
        openai_messages = service.transform_messages_for_openai(request.messages)
        logger.debug(f'Трансформированные сообщения: {openai_messages}')

        # Если запрошена потоковая передача
        if request.stream:
            logger.info('Обработка потокового запроса')
            return StreamingResponse(
                service.stream_chat_response(openai_messages, request.model),
                media_type='application/x-ndjson',
                headers={'Cache-Control': 'no-cache'},
            )

        # Обычный ответ без потоковой передачи
        response_data = await service.call_openai_api(
            openai_messages, request.model, stream=False
        )

        generated_text = service.extract_content_from_openai_response(response_data)
        logger.info(f'Получен ответ длиной {len(generated_text)} символов')

        return OllamaChatResponse(
            model=request.model,
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            message=OllamaMessage(role='assistant', content=generated_text),
            done=True,
        )
    except Exception as e:
        logger.error(f'Ошибка в chat_with_model: {e!s}', exc_info=True)
        service.raise_connection_error(e, settings.api_base_url)
        # noinspection PyUnreachableCode
        return None
