import json
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Never

from config import settings
from constants import (
    DEFAULT_MODEL,
    HTTP_OK,
    RESPONSE_PREFIX,
)
from logger import logger
from models import (
    OllamaMessage,
    OllamaModel,
    OllamaModelDetails,
    OllamaModelsResponse,
)

if TYPE_CHECKING:
    import httpx


# HTTP клиент holder
class HTTPClientHolder:
    """Контейнер для HTTP клиента, избегаем global."""

    def __init__(self) -> None:
        self.client: httpx.AsyncClient | None = None


client_holder = HTTPClientHolder()

fake_model = OllamaModel(
    name='fake_model',
    model='fake_model',
    modified_at='2024-03-15T14:30:25.123456789Z',
    size=3825819648,
    digest='sha256:4f4c8b3e5d9a2f1b8c7e6d5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c',
    details=OllamaModelDetails(
        parent_model='llama2',
        format='gguf',
        family='llama',
        families=['llama', 'chat'],
        parameter_size='7B',
        quantization_level='Q4_0',
    ),
)


class OpenAIService:
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

    def create_ollama_models_response(
        self, models: list[dict[str, Any]]
    ) -> OllamaModelsResponse:
        """Создает ответ в формате Ollama из списка моделей."""
        ollama_models = []

        if len(models) == 0:
            ollama_models.append(fake_model)
            logger.warning('Не удалось получить модели. Добавлена фейковая модель.')

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
            ollama_models.append(ollama_model)

        return OllamaModelsResponse(models=ollama_models)

    def transform_messages_for_openai(
        self,
        messages: list[OllamaMessage],
    ) -> list[dict[str, str]]:
        """Преобразует сообщения Ollama в формат OpenAI."""
        return [{'role': msg.role, 'content': msg.content} for msg in messages]

    def extract_content_from_openai_response(
        self, response_data: dict[str, Any]
    ) -> str:
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

    async def call_openai_api_stream(
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

                logo_sent = False

                async for line in response.aiter_lines():
                    clear_line = line.strip()
                    if clear_line:
                        clear_line = clear_line.removeprefix('data: ')

                        if clear_line == '[DONE]':
                            break

                        try:
                            chunk_data = json.loads(clear_line)

                            if not logo_sent:
                                logo_sent = self._add_greeting_to_chank(chunk_data)

                            yield chunk_data
                        except json.JSONDecodeError:
                            logger.warning(f'Не удалось парсить JSON: {clear_line}')
                            continue

        except Exception as e:
            self.raise_connection_error(e, str(client_holder.client.base_url))

    async def stream_chat_response(
        self, messages: list[dict[str, str]], model: str
    ) -> AsyncGenerator[str]:
        """Генерирует потоковый ответ в формате Ollama."""
        logger.info('Начинаем потоковую передачу ответа')

        accumulated_content = ''
        created_at = time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        try:
            async for chunk in self.call_openai_api_stream(messages, model):
                # Извлекаем контент из chunk'а OpenAI
                delta_content = self._extract_delta_content(chunk)

                if delta_content:
                    accumulated_content += delta_content

                    # Формируем ответ в формате Ollama
                    ollama_chunk = {
                        'model': model,
                        'created_at': created_at,
                        'message': {'role': 'assistant', 'content': delta_content},
                        'done': False,
                    }

                    # Отправляем chunk в формате NDJSON
                    yield json.dumps(ollama_chunk, ensure_ascii=False) + '\n'

            # Отправляем финальный chunk с done=True
            final_chunk = {
                'model': model,
                'created_at': created_at,
                'message': {'role': 'assistant', 'content': ''},
                'done': True,
            }

            yield json.dumps(final_chunk, ensure_ascii=False) + '\n'
            logger.info(
                f'Потоковая передача завершена. '
                f'Общая длина: {len(accumulated_content)} символов'
            )

        except Exception as e:
            logger.error(f'Ошибка в потоковой передаче: {e!s}', exc_info=True)
            # Отправляем финальный chunk с ошибкой
            error_chunk = {
                'model': model,
                'created_at': created_at,
                'message': {'role': 'assistant', 'content': f'{e!s}'},
                'done': True,
            }
            yield json.dumps(error_chunk, ensure_ascii=False) + '\n'

    def _extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Извлекает контент из delta chunk'а OpenAI."""
        try:
            choices = chunk.get('choices', [])
            if choices and isinstance(choices, list):
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    delta = first_choice.get('delta', {})
                    if isinstance(delta, dict):
                        content = delta.get('content', '')
                        if isinstance(content, str):
                            return content
        except (IndexError, KeyError, TypeError):
            pass

        return ''

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

    def _add_greeting_to_chank(self, chunk: dict, *, add_new_line: bool = True) -> bool:
        try:
            content = chunk['choices'][0]['delta']['content'] if chunk else None

            if content and (len(content) > 0):
                prefix = (
                    RESPONSE_PREFIX + ' \n\n' if add_new_line else RESPONSE_PREFIX + ' '
                )

                chunk['choices'][0]['delta']['content'] = prefix + content
            else:
                return False

        except (KeyError, IndexError, TypeError):
            return False
        else:
            return True
