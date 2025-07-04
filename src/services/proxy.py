import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from constants import (
    DEFAULT_MODEL,
    RESPONSE_PREFIX,
)
from logger import logger
from models.ollama import (
    OllamaModel,
    OllamaModelDetails,
)
from ports.spi.openai.gateway import OpenAIGateway

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
        num_parameters=7_000_000_000,
    ),
)


gateway = OpenAIGateway()


class OpenAIService:
    def __init__(self): ...

    async def call_api(
        self, messages: list[dict[str, str]], model: str, *, stream: bool = False
    ) -> str:
        response_data = await gateway.call_openai_api(messages, model, stream=stream)

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
            logger.warning('Invalid response from OpenAI', exc_info=True)
            return ''
        except Exception:
            logger.error('Error processing response from OpenAI', exc_info=True)
            raise

        return ''

    async def get_models(self) -> list[OllamaModel]:
        """Create Ollama format response from model list."""
        models = await gateway.get_models_from_api()

        ollama_models = []

        if len(models) == 0:
            ollama_models.append(fake_model)
            logger.warning('Failed to get models. Added fake model.')

        for model in models:
            model_id = model.get('id', DEFAULT_MODEL)
            created_time_str = model.get('created', str(int(time.time())))

            try:
                created_time = (
                    int(created_time_str)
                    if isinstance(created_time_str, str)
                    else created_time_str
                )
            except (ValueError, TypeError):
                created_time = int(time.time())

            # Create model in Ollama format
            ollama_model = OllamaModel(
                name=model_id,
                model=model_id,
                modified_at=time.strftime(
                    '%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(float(created_time))
                ),
                size=1000000000,  # 1GB approximate size
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

        return ollama_models

    async def get_stream(
        self, messages: list[dict[str, str]], model: str
    ) -> AsyncGenerator[str]:
        """Generate streaming response in Ollama format."""
        logger.info('Starting streaming response')

        accumulated_content = ''
        created_at = time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        try:
            logo_sent = False

            async for chunk in gateway.get_stream(messages, model):
                if not logo_sent:
                    logo_sent = self._add_greeting(chunk)

                delta_content = self._extract_delta_content(chunk)

                if delta_content:
                    accumulated_content += delta_content

                    ollama_chunk = {
                        'model': model,
                        'created_at': created_at,
                        'message': {'role': 'assistant', 'content': delta_content},
                        'done': False,
                    }

                    yield json.dumps(ollama_chunk, ensure_ascii=False) + '\n'

            final_chunk = {
                'model': model,
                'created_at': created_at,
                'message': {'role': 'assistant', 'content': ''},
                'done': True,
            }

            yield json.dumps(final_chunk, ensure_ascii=False) + '\n'
            logger.info(
                f'Streaming completed. '
                f'Total length: {len(accumulated_content)} characters'
            )

        except Exception as e:
            logger.error(f'Error in streaming: {e!s}', exc_info=True)

            error_chunk = {
                'model': model,
                'created_at': created_at,
                'message': {'role': 'assistant', 'content': f'{e!s}'},
                'done': True,
            }
            yield json.dumps(error_chunk, ensure_ascii=False) + '\n'

    @staticmethod
    def _extract_delta_content(chunk: dict[str, Any]) -> str:
        """Extract content from OpenAI delta chunk."""
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

    @staticmethod
    def _add_greeting(chunk: dict, *, add_new_line: bool = True) -> bool:
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
