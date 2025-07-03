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


# This endpoint is checked by AI Assistant on a first connection
@root_router.get('/')
async def health_check() -> OllamaHealthResponse:
    """Health check endpoint."""
    return OllamaHealthResponse()


def error_handler[F: Callable[..., Awaitable[Any]]](func: F) -> F:
    """Handle errors in endpoints."""

    @wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTPException as is
            raise
        except Exception as e:
            logger.exception('Unexpected error in endpoint')
            raise HTTPException(
                status_code=HTTP_BAD_GATEWAY_ERROR, detail=str(e)
            ) from e

    return wrapper  # type: ignore[return-value]


@ollama_router.get('/version')
@error_handler
async def get_version() -> OllamaVersionResponse:
    """Return API version."""
    return OllamaVersionResponse(version=API_VERSION)


@ollama_router.get('/tags')
@error_handler
@cache(expire=CACHE_EXPIRATION_S)
async def tags(service: OpenAIServiceDep) -> OllamaModelsResponse | None:
    """Return list of available models."""
    logger.info('Received request to /api/tags')

    models = await service.get_models()

    logger.info(f'Received {len(models) if models else 0} models')

    return OllamaModelsResponse(models=models)


@ollama_router.post('/show')
@error_handler
async def show(request: OllamaShowRequest) -> OllamaShowResponse:
    """Return model information."""
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
    """Simulate model download."""
    return OllamaPullResponse(
        status=f'pulling {request.name}', total=1000000000, completed=1000000000
    )


@ollama_router.post('/generate')
@error_handler
async def generate(
    request: OllamaGenerateRequest, service: OpenAIServiceDep
) -> OllamaGenerateResponse:
    """Generate text using external API."""
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
    """Chat with model using external API."""
    # Add detailed logging
    logger.info(
        f'Received request to /api/chat: model={request.model}, '
        f'messages_count={len(request.messages)}, stream={request.stream}'
    )
    logger.debug(f'Full request: {request.model_dump()}')

    try:
        openai_messages = [
            {'role': msg.role, 'content': msg.content} for msg in request.messages
        ]

        logger.debug(f'Transformed messages: {openai_messages}')

        # If streaming is requested
        if request.stream:
            logger.info('Processing streaming request')

            return StreamingResponse(
                service.get_stream(messages=openai_messages, model=request.model),
                media_type='application/x-ndjson',
                headers={'Cache-Control': 'no-cache'},
            )

        # Regular response without streaming
        generated_text = await service.call_api(
            openai_messages, request.model, stream=False
        )

        logger.info(f'Received response with {len(generated_text)} characters')

        return OllamaChatResponse(
            model=request.model,
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            message=OllamaMessage(role='assistant', content=generated_text),
            done=True,
        )
    except Exception as e:
        logger.error(f'Error in chat_with_model: {e!s}', exc_info=True)

        raise HTTPException(status_code=HTTP_BAD_GATEWAY_ERROR, detail=str(e)) from e
