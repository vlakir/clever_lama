import json
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Never

from config import settings
from constants import DEFAULT_MODEL, HTTP_OK, RESPONSE_PREFIX
from logger import logger

if TYPE_CHECKING:
    import httpx


# HTTP клиент holder
class HTTPClientHolder:
    """Контейнер для HTTP клиента, избегаем global."""

    def __init__(self) -> None:
        self.client: httpx.AsyncClient | None = None


client_holder = HTTPClientHolder()


class OpenAIGateway:
    def __init__(self): ...

    async def health_check_external_api(self) -> None:
        """Проверяет доступность внешнего API."""
        if not client_holder.client:
            msg = 'HTTP клиент не инициализирован'
            self.raise_connection_error(Exception(msg), settings.api_base_url)
        try:
            response = await client_holder.client.get('/models')
            if response.status_code == HTTP_OK:
                logger.info('✅ Внешний API доступен')
            else:
                logger.warning(f'⚠️ Внешний API вернул статус {response.status_code}')
        except Exception:
            logger.error(f'❌ Внешний API {settings.api_base_url} недоступен')

    async def get_models_from_api(self) -> list[dict[str, Any]]:
        """Получает список моделей из внешнего API."""
        models = []

        try:
            if client_holder.client is not None:
                response = await client_holder.client.get('/models')
                response.raise_for_status()

                data = response.json()

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
            logger.error('Ошибка получения моделей из API')

        return models

    async def call_openai_api(
        self, messages: list[dict[str, str]], model: str, *, stream: bool = False
    ) -> dict[str, Any]:
        """Вызывает OpenAI API для генерации ответа."""
        if not client_holder.client:
            msg = 'HTTP клиент не инициализирован'
            self.raise_connection_error(Exception(msg), settings.api_base_url)

        try:
            payload = {
                'model': model,
                'messages': messages,
                'stream': stream,
            }

            response = await client_holder.client.post(
                '/chat/completions', json=payload
            )
            response.raise_for_status()

        except Exception as e:
            self.raise_connection_error(e, settings.api_base_url)

        else:
            return response.json()

    async def get_stream(
        self, messages: list[dict[str, str]], model: str
    ) -> AsyncGenerator[dict[str, Any]]:
        """Вызывает OpenAI API для потоковой генерации ответа."""
        if not client_holder.client:
            msg = 'HTTP клиент не инициализирован'
            self.raise_connection_error(Exception(msg), settings.api_base_url)

        try:
            payload = {
                'model': model,
                'messages': messages,
                'stream': True,
            }

            async with client_holder.client.stream(
                'POST', '/chat/completions', json=payload
            ) as response:
                response.raise_for_status()

                # logo_sent = False

                async for line in response.aiter_lines():
                    clear_line = line.strip()
                    if clear_line:
                        clear_line = clear_line.removeprefix('data: ')

                        if clear_line == '[DONE]':
                            break

                        try:
                            chunk = json.loads(clear_line)
                            #
                            # if not logo_sent:
                            #     logo_sent = self._add_greeting(chunk)

                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning(f'Не удалось парсить JSON: {clear_line}')
                            continue

        except Exception as e:
            self.raise_connection_error(e, str(client_holder.client.base_url))

    def raise_connection_error(
        self, exception: Exception, gateway_url: str, *, add_new_line: bool = False
    ) -> Never:
        logger.error('Ошибка вызова OpenAI')

        prefix = RESPONSE_PREFIX + ' \n\n' if add_new_line else RESPONSE_PREFIX

        msg = (
            f'{prefix} Ошибка подключения к OpenAI API по '
            f'адресу {gateway_url}. Проверьте правильность '
            f'адреса или наличие стабильного '
            f'интернет-подключения.'
        )
        raise ConnectionAbortedError(msg) from exception
