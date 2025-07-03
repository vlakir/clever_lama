"""Tests for the OpenAIGateway class."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from clever_lama.ports.spi.openai.gateway import OpenAIGateway, HTTPClientHolder, client_holder


class TestHTTPClientHolder:
    """Test cases for HTTPClientHolder class."""

    def test_init(self):
        """Test HTTPClientHolder initialization."""
        holder = HTTPClientHolder()
        assert holder.client is None

    def test_client_assignment(self):
        """Test client assignment."""
        holder = HTTPClientHolder()
        mock_client = MagicMock()
        holder.client = mock_client
        assert holder.client is mock_client


class TestOpenAIGateway:
    """Test cases for OpenAIGateway class."""

    @pytest.fixture
    def gateway(self):
        """Create OpenAIGateway instance for testing."""
        return OpenAIGateway()

    @pytest.fixture
    def mock_client(self):
        """Mock HTTP client for testing."""
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture(autouse=True)
    def setup_client_holder(self, mock_client):
        """Setup client holder with mock client."""
        original_client = client_holder.client
        client_holder.client = mock_client
        yield
        client_holder.client = original_client

    @pytest.mark.asyncio
    async def test_health_check_external_api_success(self, gateway, mock_client):
        """Test health_check_external_api with successful response."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        # Act
        await gateway.health_check_external_api()

        # Assert
        mock_client.get.assert_called_once_with('/models')

    @pytest.mark.asyncio
    async def test_health_check_external_api_bad_status(self, gateway, mock_client):
        """Test health_check_external_api with bad status code."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.get.return_value = mock_response

        # Act
        await gateway.health_check_external_api()

        # Assert
        mock_client.get.assert_called_once_with('/models')

    @pytest.mark.asyncio
    async def test_health_check_external_api_exception(self, gateway, mock_client):
        """Test health_check_external_api with exception."""
        # Arrange
        mock_client.get.side_effect = Exception("Connection error")

        # Act
        await gateway.health_check_external_api()

        # Assert
        mock_client.get.assert_called_once_with('/models')

    @pytest.mark.asyncio
    async def test_health_check_external_api_no_client(self, gateway):
        """Test health_check_external_api with no client."""
        # Arrange
        client_holder.client = None

        # Act & Assert
        with pytest.raises(ConnectionAbortedError):
            await gateway.health_check_external_api()

    @pytest.mark.asyncio
    async def test_get_models_from_api_success(self, gateway, mock_client):
        """Test get_models_from_api with successful response."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model1", "object": "model"},
                {"id": "model2", "object": "model"}
            ]
        }
        mock_client.get.return_value = mock_response

        # Act
        result = await gateway.get_models_from_api()

        # Assert
        assert len(result) == 2
        assert result[0]["id"] == "model1"
        assert result[1]["id"] == "model2"
        assert all(model["owned_by"] == "aitunnel" for model in result)
        mock_client.get.assert_called_once_with('/models')
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_models_from_api_no_data(self, gateway, mock_client):
        """Test get_models_from_api with no data in response."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_client.get.return_value = mock_response

        # Act
        result = await gateway.get_models_from_api()

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_get_models_from_api_exception(self, gateway, mock_client):
        """Test get_models_from_api with exception."""
        # Arrange
        mock_client.get.side_effect = Exception("API Error")

        # Act
        result = await gateway.get_models_from_api()

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_get_models_from_api_no_client(self, gateway):
        """Test get_models_from_api with no client."""
        # Arrange
        client_holder.client = None

        # Act
        result = await gateway.get_models_from_api()

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_call_openai_api_success(self, gateway, mock_client):
        """Test call_openai_api with successful response."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        expected_response = {"choices": [{"message": {"content": "Hi there!"}}]}

        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_client.post.return_value = mock_response

        # Act
        result = await gateway.call_openai_api(messages, model, stream=False)

        # Assert
        assert result == expected_response
        mock_client.post.assert_called_once_with(
            '/chat/completions',
            json={
                'model': model,
                'messages': messages,
                'stream': False
            }
        )
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_openai_api_with_stream(self, gateway, mock_client):
        """Test call_openai_api with streaming enabled."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        expected_response = {"choices": [{"delta": {"content": "Hi"}}]}

        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_client.post.return_value = mock_response

        # Act
        result = await gateway.call_openai_api(messages, model, stream=True)

        # Assert
        assert result == expected_response
        mock_client.post.assert_called_once_with(
            '/chat/completions',
            json={
                'model': model,
                'messages': messages,
                'stream': True
            }
        )

    @pytest.mark.asyncio
    async def test_call_openai_api_no_client(self, gateway):
        """Test call_openai_api with no client."""
        # Arrange
        client_holder.client = None
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"

        # Act & Assert
        with pytest.raises(ConnectionAbortedError):
            await gateway.call_openai_api(messages, model)

    @pytest.mark.asyncio
    async def test_call_openai_api_exception(self, gateway, mock_client):
        """Test call_openai_api with exception."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        mock_client.post.side_effect = Exception("API Error")

        # Act & Assert
        with pytest.raises(ConnectionAbortedError):
            await gateway.call_openai_api(messages, model)

    @pytest.mark.asyncio
    async def test_get_stream_success(self, gateway, mock_client):
        """Test get_stream with successful response."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"

        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        class MockAsyncIterator:
            def __init__(self, lines):
                self.lines = lines
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.lines):
                    raise StopAsyncIteration
                line = self.lines[self.index]
                self.index += 1
                return line

        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            'data: [DONE]'
        ]
        mock_response.aiter_lines = lambda: MockAsyncIterator(lines)
        mock_client.stream.return_value.__aenter__.return_value = mock_response

        # Act
        result_chunks = []
        async for chunk in gateway.get_stream(messages, model):
            result_chunks.append(chunk)

        # Assert
        assert len(result_chunks) == 2
        assert result_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert result_chunks[1]["choices"][0]["delta"]["content"] == " world"

        mock_client.stream.assert_called_once_with(
            'POST',
            '/chat/completions',
            json={
                'model': model,
                'messages': messages,
                'stream': True
            }
        )

    @pytest.mark.asyncio
    async def test_get_stream_invalid_json(self, gateway, mock_client):
        """Test get_stream with invalid JSON."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        class MockAsyncIterator:
            def __init__(self, lines):
                self.lines = lines
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.lines):
                    raise StopAsyncIteration
                line = self.lines[self.index]
                self.index += 1
                return line

        lines = [
            'data: invalid json',
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: [DONE]'
        ]
        mock_response.aiter_lines = lambda: MockAsyncIterator(lines)
        mock_client.stream.return_value.__aenter__.return_value = mock_response

        # Act
        result_chunks = []
        async for chunk in gateway.get_stream(messages, model):
            result_chunks.append(chunk)

        # Assert
        assert len(result_chunks) == 1  # Only valid JSON should be yielded
        assert result_chunks[0]["choices"][0]["delta"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_stream_no_client(self, gateway):
        """Test get_stream with no client."""
        # Arrange
        client_holder.client = None
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"

        # Act & Assert
        with pytest.raises(ConnectionAbortedError):
            async for _ in gateway.get_stream(messages, model):
                pass

    @pytest.mark.asyncio
    async def test_get_stream_exception(self, gateway, mock_client):
        """Test get_stream with exception."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "test-model"
        mock_client.stream.side_effect = Exception("Stream Error")

        # Act & Assert
        with pytest.raises(ConnectionAbortedError):
            async for _ in gateway.get_stream(messages, model):
                pass

    def test_raise_connection_error_basic(self, gateway):
        """Test raise_connection_error method."""
        # Arrange
        exception = Exception("Test error")
        gateway_url = "https://api.example.com"

        # Act & Assert
        with pytest.raises(ConnectionAbortedError) as exc_info:
            gateway.raise_connection_error(exception, gateway_url)

        assert "Ошибка подключения к OpenAI API" in str(exc_info.value)
        assert gateway_url in str(exc_info.value)

    def test_raise_connection_error_with_newline(self, gateway):
        """Test raise_connection_error method with newline."""
        # Arrange
        exception = Exception("Test error")
        gateway_url = "https://api.example.com"

        # Act & Assert
        with pytest.raises(ConnectionAbortedError) as exc_info:
            gateway.raise_connection_error(exception, gateway_url, add_new_line=True)

        error_message = str(exc_info.value)
        assert "Ошибка подключения к OpenAI API" in error_message
        assert "\n\n" in error_message
