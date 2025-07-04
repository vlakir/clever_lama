"""Tests for the OpenAIService proxy service."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from services.proxy import OpenAIService
from models.ollama import OllamaModel


class TestOpenAIService:
    """Test cases for OpenAIService class."""

    @pytest.fixture
    def service(self):
        """Create OpenAIService instance for testing."""
        return OpenAIService()

    @pytest.fixture
    def mock_gateway(self):
        """Mock OpenAIGateway for testing."""
        with patch('services.proxy.gateway') as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_call_api_non_stream(self, service, mock_gateway):
        """Test call_api method with non-streaming response."""
        
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        expected_response = {
            "choices": [{"message": {"content": "Hello, how can I help you?"}}]
        }
        mock_gateway.call_openai_api = AsyncMock(return_value=expected_response)

        
        result = await service.call_api(messages, model, stream=False)

        
        assert result == "Hello, how can I help you?"
        mock_gateway.call_openai_api.assert_called_once_with(
            messages, model, stream=False
        )

    @pytest.mark.asyncio
    async def test_call_api_stream(self, service, mock_gateway):
        """Test call_api method with streaming response."""
        
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        expected_response = {
            "choices": [{"message": {"content": "Hello, how can I help you?"}}]
        }
        mock_gateway.call_openai_api = AsyncMock(return_value=expected_response)

        
        result = await service.call_api(messages, model, stream=True)

        
        assert result == "Hello, how can I help you?"
        mock_gateway.call_openai_api.assert_called_once_with(messages, model, stream=True)

    @pytest.mark.asyncio
    async def test_call_api_empty_response(self, service, mock_gateway):
        """Test call_api method with empty response."""
        
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        mock_gateway.call_openai_api = AsyncMock(return_value={"choices": []})

        
        result = await service.call_api(messages, model, stream=False)

        
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_models_success(self, service, mock_gateway):
        """Test get_models method with successful response."""
        
        api_models = [
            {"id": "model1", "object": "model", "created": "123456", "owned_by": "test"},
            {"id": "model2", "object": "model", "created": "123457", "owned_by": "test"}
        ]
        mock_gateway.get_models_from_api = AsyncMock(return_value=api_models)

        
        result = await service.get_models()

        
        assert len(result) == 2
        assert all(isinstance(model, OllamaModel) for model in result)
        assert result[0].name == "model1"
        assert result[1].name == "model2"

    @pytest.mark.asyncio
    async def test_get_models_empty_response(self, service, mock_gateway):
        """Test get_models method with empty response."""
        
        mock_gateway.get_models_from_api = AsyncMock(return_value=[])

        
        result = await service.get_models()

        
        assert len(result) == 1  # Should return fake_model when no models available
        assert result[0].name == "fake_model"

    @pytest.mark.asyncio
    async def test_get_stream(self, service, mock_gateway):
        """Test get_stream method."""
        
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"

        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]}
            ]
            for chunk in chunks:
                yield chunk

        mock_gateway.get_stream.return_value = mock_stream()

        
        result_generator = service.get_stream(messages, model)
        results = []
        async for chunk in result_generator:
            results.append(chunk)

        
        assert len(results) == 3  # 2 content chunks + 1 final chunk
        # Check that the first chunk contains greeting
        first_chunk = json.loads(results[0])
        assert "🦙" in first_chunk["message"]["content"]

    def test_extract_delta_content_with_content(self, service):
        """Test _extract_delta_content with content."""
        
        chunk = {"choices": [{"delta": {"content": "test content"}}]}

        
        result = service._extract_delta_content(chunk)

        
        assert result == "test content"

    def test_extract_delta_content_without_content(self, service):
        """Test _extract_delta_content without content."""
        
        chunk = {"choices": [{"delta": {}}]}

        
        result = service._extract_delta_content(chunk)

        
        assert result == ""

    def test_extract_delta_content_empty_choices(self, service):
        """Test _extract_delta_content with empty choices."""
        
        chunk = {"choices": []}

        
        result = service._extract_delta_content(chunk)

        
        assert result == ""

    def test_add_greeting_with_newline(self, service):
        """Test _add_greeting method with newline."""
        
        chunk = {"choices": [{"delta": {"content": "original content"}}]}

        
        result = service._add_greeting(chunk, add_new_line=True)

        
        assert result is True
        expected_content = "🦙 \n\noriginal content"
        assert chunk["choices"][0]["delta"]["content"] == expected_content

    def test_add_greeting_without_newline(self, service):
        """Test _add_greeting method without newline."""
        
        chunk = {"choices": [{"delta": {"content": "original content"}}]}

        
        result = service._add_greeting(chunk, add_new_line=False)

        
        assert result is True
        expected_content = "🦙 original content"
        assert chunk["choices"][0]["delta"]["content"] == expected_content

    @pytest.mark.asyncio
    async def test_call_api_exception_handling(self, service, mock_gateway):
        """Test call_api method exception handling."""
        
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        mock_gateway.call_openai_api.side_effect = Exception("API Error")

         
        with pytest.raises(Exception, match="API Error"):
            await service.call_api(messages, model, stream=False)

    @pytest.mark.asyncio
    async def test_get_models_exception_handling(self, service, mock_gateway):
        """Test get_models method exception handling."""
        
        mock_gateway.get_models_from_api = AsyncMock(return_value=[])  # Exception results in empty list

        
        result = await service.get_models()

        
        assert len(result) == 1  # Should return fake_model when exception occurs
        assert result[0].name == "fake_model"
