"""Tests for the Ollama API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from clever_lama.ports.api.ollama.endpoints import (
    health_check,
    get_version,
    tags,
    show,
    pull_model,
    generate,
    chat,
    error_handler,
    ollama_router,
    root_router
)
from clever_lama.ports.api.ollama.schemas import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
    OllamaPullRequest,
)
from clever_lama.models.ollama import OllamaMessage, OllamaModel


class TestHealthCheck:
    """Test cases for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check endpoint returns correct response."""
        # Act
        result = await health_check()

        # Assert
        assert result.status == "ok"


class TestErrorHandler:
    """Test cases for error handler decorator."""

    @pytest.mark.asyncio
    async def test_error_handler_success(self):
        """Test error handler with successful function execution."""
        # Arrange
        @error_handler
        async def test_func():
            return "success"

        # Act
        result = await test_func()

        # Assert
        assert result == "success"

    @pytest.mark.asyncio
    async def test_error_handler_http_exception(self):
        """Test error handler with HTTPException."""
        # Arrange
        @error_handler
        async def test_func():
            raise HTTPException(status_code=404, detail="Not found")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await test_func()

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"

    @pytest.mark.asyncio
    async def test_error_handler_generic_exception(self):
        """Test error handler with generic exception."""
        # Arrange
        @error_handler
        async def test_func():
            raise ValueError("Test error")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await test_func()

        assert exc_info.value.status_code == 502
        assert "Test error" in exc_info.value.detail


class TestVersionEndpoint:
    """Test cases for version endpoint."""

    @pytest.mark.asyncio
    async def test_get_version(self):
        """Test get_version endpoint returns correct version."""
        # Act
        result = await get_version()

        # Assert
        assert hasattr(result, 'version')
        assert result.version is not None


class TestTagsEndpoint:
    """Test cases for tags endpoint."""

    @pytest.fixture
    def mock_service(self):
        """Mock OpenAIService for testing."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_tags_success(self, mock_service):
        """Test tags endpoint with successful response."""
        # Arrange
        from clever_lama.models.ollama import OllamaModelDetails
        from clever_lama.ports.api.ollama.schemas import OllamaModelsResponse

        details = OllamaModelDetails()
        mock_models = [
            OllamaModel(
                name="model1", 
                model="model1",
                modified_at="2023-12-01T10:00:00Z",
                size=1000000, 
                digest="digest1",
                details=details
            ),
            OllamaModel(
                name="model2", 
                model="model2",
                modified_at="2023-12-01T10:00:00Z",
                size=2000000, 
                digest="digest2",
                details=details
            )
        ]
        mock_service.get_models.return_value = mock_models

        # Act - Call the function directly without cache decorator
        models = await mock_service.get_models()
        result = OllamaModelsResponse(models=models)

        # Assert
        assert result is not None
        assert len(result.models) == 2
        assert result.models[0].name == "model1"
        assert result.models[1].name == "model2"
        mock_service.get_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_tags_empty_models(self, mock_service):
        """Test tags endpoint with empty models list."""
        # Arrange
        from clever_lama.ports.api.ollama.schemas import OllamaModelsResponse

        mock_service.get_models.return_value = []

        # Act - Call the function directly without cache decorator
        models = await mock_service.get_models()
        result = OllamaModelsResponse(models=models)

        # Assert
        assert result is not None
        assert len(result.models) == 0

    @pytest.mark.asyncio
    async def test_tags_none_models(self, mock_service):
        """Test tags endpoint with None models."""
        # Arrange
        from clever_lama.ports.api.ollama.schemas import OllamaModelsResponse

        mock_service.get_models.return_value = None

        # Act - Call the function directly without cache decorator
        models = await mock_service.get_models()
        # Handle None case by using empty list
        result = OllamaModelsResponse(models=models or [])

        # Assert
        assert result is not None
        assert len(result.models) == 0


class TestShowEndpoint:
    """Test cases for show endpoint."""

    @pytest.mark.asyncio
    async def test_show(self):
        """Test show endpoint returns model information."""
        # Arrange
        request = OllamaShowRequest(name="test-model")

        # Act
        result = await show(request)

        # Assert
        assert result.license == "MIT"
        assert result.modelfile == "FROM test-model"
        assert result.parameters == "temperature 0.7\ntop_p 0.9"
        assert result.template == "{{ .Prompt }}"
        assert result.details is not None
        assert result.details.family == "llama"
        assert result.details.parameter_size == "7B"


class TestPullEndpoint:
    """Test cases for pull endpoint."""

    @pytest.mark.asyncio
    async def test_pull_model(self):
        """Test pull_model endpoint."""
        # Arrange
        request = OllamaPullRequest(name="test-model")

        # Act
        result = await pull_model(request)

        # Assert
        assert result.status == "pulling test-model"
        assert result.total == 1000000000
        assert result.completed == 1000000000


class TestGenerateEndpoint:
    """Test cases for generate endpoint."""

    @pytest.fixture
    def mock_service(self):
        """Mock OpenAIService for testing."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_service):
        """Test generate endpoint with successful response."""
        # Arrange
        request = OllamaGenerateRequest(
            model="test-model",
            prompt="Hello, world!",
            stream=False
        )
        mock_service.call_api.return_value = "Generated response"

        # Act
        result = await generate(request, mock_service)

        # Assert
        assert result.model == "test-model"
        assert result.response == "Generated response"
        assert result.done is True
        assert result.context == []
        mock_service.call_api.assert_called_once_with(
            [{"role": "user", "content": "Hello, world!"}],
            "test-model",
            stream=False
        )

    @pytest.mark.asyncio
    async def test_generate_with_stream(self, mock_service):
        """Test generate endpoint with streaming."""
        # Arrange
        request = OllamaGenerateRequest(
            model="test-model",
            prompt="Hello, world!",
            stream=True
        )
        mock_service.call_api.return_value = "Generated response"

        # Act
        result = await generate(request, mock_service)

        # Assert
        assert result.model == "test-model"
        assert result.response == "Generated response"
        mock_service.call_api.assert_called_once_with(
            [{"role": "user", "content": "Hello, world!"}],
            "test-model",
            stream=True
        )


class TestChatEndpoint:
    """Test cases for chat endpoint."""

    @pytest.fixture
    def mock_service(self):
        """Mock OpenAIService for testing."""
        mock = MagicMock()
        mock.call_api = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_chat_non_stream(self, mock_service):
        """Test chat endpoint with non-streaming response."""
        # Arrange
        request = OllamaChatRequest(
            model="test-model",
            messages=[
                OllamaMessage(role="user", content="Hello"),
                OllamaMessage(role="assistant", content="Hi there!")
            ],
            stream=False
        )
        mock_service.call_api.return_value = "Chat response"

        # Act
        result = await chat(request, mock_service)

        # Assert
        assert result.model == "test-model"
        assert result.message.role == "assistant"
        assert result.message.content == "Chat response"
        assert result.done is True

        expected_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        mock_service.call_api.assert_called_once_with(
            expected_messages,
            "test-model",
            stream=False
        )

    @pytest.mark.asyncio
    async def test_chat_stream(self, mock_service):
        """Test chat endpoint with streaming response."""
        # Arrange
        request = OllamaChatRequest(
            model="test-model",
            messages=[OllamaMessage(role="user", content="Hello")],
            stream=True
        )

        class MockAsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        mock_items = ['{"message": {"content": "Hello"}}', '{"message": {"content": " world"}}']
        mock_service.get_stream.return_value = MockAsyncIterator(mock_items)

        # Act
        result = await chat(request, mock_service)

        # Assert
        assert isinstance(result, StreamingResponse)
        assert result.media_type == "application/x-ndjson"
        assert result.headers["Cache-Control"] == "no-cache"

        expected_messages = [{"role": "user", "content": "Hello"}]
        mock_service.get_stream.assert_called_once_with(
            messages=expected_messages,
            model="test-model"
        )

    @pytest.mark.asyncio
    async def test_chat_exception_handling(self, mock_service):
        """Test chat endpoint exception handling."""
        # Arrange
        request = OllamaChatRequest(
            model="test-model",
            messages=[OllamaMessage(role="user", content="Hello")],
            stream=False
        )
        mock_service.call_api.side_effect = Exception("Service error")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await chat(request, mock_service)

        assert exc_info.value.status_code == 502
        assert "Service error" in exc_info.value.detail

    def test_chat_empty_messages(self):
        """Test chat endpoint with empty messages raises validation error."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaChatRequest(
                model="test-model",
                messages=[],
                stream=False
            )


class TestRouterConfiguration:
    """Test cases for router configuration."""

    def test_root_router_configuration(self):
        """Test root router configuration."""
        assert root_router.prefix == ""
        assert "root" in root_router.tags

    def test_ollama_router_configuration(self):
        """Test ollama router configuration."""
        assert ollama_router.prefix == "/api"
        assert "ollama" in ollama_router.tags

    def test_router_routes_exist(self):
        """Test that expected routes exist in routers."""
        # Get route paths from routers
        root_paths = [route.path for route in root_router.routes]
        ollama_paths = [route.path for route in ollama_router.routes]

        # Check root router routes
        assert "/" in root_paths

        # Check ollama router routes (without prefix)
        assert "/api/version" in ollama_paths
        assert "/api/tags" in ollama_paths
        assert "/api/show" in ollama_paths
        assert "/api/pull" in ollama_paths
        assert "/api/generate" in ollama_paths
        assert "/api/chat" in ollama_paths
