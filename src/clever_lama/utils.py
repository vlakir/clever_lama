from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from fastapi import HTTPException

from src.clever_lama.constants import HTTP_SERVER_ERROR, SERVER_ERROR_MESSAGE
from src.clever_lama.logger import logger
from src.clever_lama.models import OllamaErrorResponse


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
                detail=OllamaErrorResponse(
                    detail=f'{SERVER_ERROR_MESSAGE}: {e}',
                    error_code='INTERNAL_ERROR',
                ).model_dump(),
            ) from e

    return wrapper  # type: ignore[return-value]
