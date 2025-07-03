"""Tests for Ollama API schemas."""

import pytest
from pydantic import ValidationError

from clever_lama.ports.api.ollama.schemas import (
    OllamaGenerateRequest,
    OllamaChatRequest,
    OllamaShowRequest,
    OllamaPullRequest,
    OllamaModelsResponse,
    OllamaShowResponse,
    OllamaGenerateResponse,
    OllamaChatResponse,
    OllamaPullResponse,
    OllamaHealthResponse,
    OllamaVersionResponse,
    OllamaErrorResponse,
)
from clever_lama.models.ollama import OllamaMessage, OllamaModel, OllamaModelDetails, OllamaOptions


class TestOllamaGenerateRequest:
    """Test cases for OllamaGenerateRequest schema."""

    def test_valid_generate_request(self):
        """Test creating a valid generate request."""
        # Act
        request = OllamaGenerateRequest(
            model="llama2:7b",
            prompt="Hello, world!",
            stream=False
        )

        # Assert
        assert request.model == "llama2:7b"
        assert request.prompt == "Hello, world!"
        assert request.stream is False
        assert request.options is None

    def test_generate_request_with_options(self):
        """Test generate request with options."""
        # Arrange
        options = OllamaOptions(temperature=0.7, max_tokens=100)

        # Act
        request = OllamaGenerateRequest(
            model="llama2:7b",
            prompt="Hello, world!",
            stream=True,
            options=options
        )

        # Assert
        assert request.model == "llama2:7b"
        assert request.prompt == "Hello, world!"
        assert request.stream is True
        assert request.options.temperature == 0.7
        assert request.options.max_tokens == 100

    def test_generate_request_default_stream(self):
        """Test generate request with default stream value."""
        # Act
        request = OllamaGenerateRequest(
            model="llama2:7b",
            prompt="Hello, world!"
        )

        # Assert
        assert request.stream is False

    def test_generate_request_empty_model(self):
        """Test validation error for empty model."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaGenerateRequest(model="", prompt="Hello, world!")

    def test_generate_request_empty_prompt(self):
        """Test validation error for empty prompt."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaGenerateRequest(model="llama2:7b", prompt="")

    def test_generate_request_missing_required_fields(self):
        """Test validation error for missing required fields."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaGenerateRequest(model="llama2:7b")  # Missing prompt

        with pytest.raises(ValidationError):
            OllamaGenerateRequest(prompt="Hello, world!")  # Missing model


class TestOllamaChatRequest:
    """Test cases for OllamaChatRequest schema."""

    def test_valid_chat_request(self):
        """Test creating a valid chat request."""
        # Arrange
        messages = [
            OllamaMessage(role="user", content="Hello"),
            OllamaMessage(role="assistant", content="Hi there!")
        ]

        # Act
        request = OllamaChatRequest(
            model="llama2:7b",
            messages=messages,
            stream=False
        )

        # Assert
        assert request.model == "llama2:7b"
        assert len(request.messages) == 2
        assert request.messages[0].role == "user"
        assert request.messages[1].role == "assistant"
        assert request.stream is False

    def test_chat_request_with_options(self):
        """Test chat request with options."""
        # Arrange
        messages = [OllamaMessage(role="user", content="Hello")]
        options = OllamaOptions(temperature=0.8, top_p=0.9)

        # Act
        request = OllamaChatRequest(
            model="llama2:7b",
            messages=messages,
            stream=True,
            options=options
        )

        # Assert
        assert request.options.temperature == 0.8
        assert request.options.top_p == 0.9

    def test_chat_request_default_stream(self):
        """Test chat request with default stream value."""
        # Arrange
        messages = [OllamaMessage(role="user", content="Hello")]

        # Act
        request = OllamaChatRequest(
            model="llama2:7b",
            messages=messages
        )

        # Assert
        assert request.stream is False

    def test_chat_request_empty_messages(self):
        """Test validation error for empty messages list."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaChatRequest(model="llama2:7b", messages=[])

    def test_chat_request_empty_model(self):
        """Test validation error for empty model."""
        # Arrange
        messages = [OllamaMessage(role="user", content="Hello")]

        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaChatRequest(model="", messages=messages)

    def test_chat_request_missing_required_fields(self):
        """Test validation error for missing required fields."""
        messages = [OllamaMessage(role="user", content="Hello")]

        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaChatRequest(model="llama2:7b")  # Missing messages

        with pytest.raises(ValidationError):
            OllamaChatRequest(messages=messages)  # Missing model


class TestOllamaShowRequest:
    """Test cases for OllamaShowRequest schema."""

    def test_valid_show_request(self):
        """Test creating a valid show request."""
        # Act
        request = OllamaShowRequest(name="llama2:7b")

        # Assert
        assert request.name == "llama2:7b"

    def test_show_request_empty_name(self):
        """Test validation error for empty name."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaShowRequest(name="")

    def test_show_request_missing_name(self):
        """Test validation error for missing name."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaShowRequest()


class TestOllamaPullRequest:
    """Test cases for OllamaPullRequest schema."""

    def test_valid_pull_request(self):
        """Test creating a valid pull request."""
        # Act
        request = OllamaPullRequest(name="llama2:7b", stream=False)

        # Assert
        assert request.name == "llama2:7b"
        assert request.stream is False

    def test_pull_request_default_stream(self):
        """Test pull request with default stream value."""
        # Act
        request = OllamaPullRequest(name="llama2:7b")

        # Assert
        assert request.stream is True  # Default is True for pull

    def test_pull_request_empty_name(self):
        """Test validation error for empty name."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaPullRequest(name="")

    def test_pull_request_missing_name(self):
        """Test validation error for missing name."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaPullRequest()


class TestOllamaModelsResponse:
    """Test cases for OllamaModelsResponse schema."""

    def test_valid_models_response(self):
        """Test creating a valid models response."""
        # Arrange
        details = OllamaModelDetails(family="llama", parameter_size="7B")
        models = [
            OllamaModel(
                name="llama2:7b",
                model="llama2:7b",
                modified_at="2023-12-01T10:00:00Z",
                size=3_800_000_000,
                digest="sha256:abc123",
                details=details
            )
        ]

        # Act
        response = OllamaModelsResponse(models=models)

        # Assert
        assert len(response.models) == 1
        assert response.models[0].name == "llama2:7b"

    def test_empty_models_response(self):
        """Test models response with empty list."""
        # Act
        response = OllamaModelsResponse(models=[])

        # Assert
        assert len(response.models) == 0


class TestOllamaShowResponse:
    """Test cases for OllamaShowResponse schema."""

    def test_valid_show_response(self):
        """Test creating a valid show response."""
        # Arrange
        details = OllamaModelDetails(family="llama", parameter_size="7B")

        # Act
        response = OllamaShowResponse(
            license="MIT",
            modelfile="FROM llama2:7b",
            parameters="temperature 0.7",
            template="{{ .Prompt }}",
            details=details
        )

        # Assert
        assert response.license == "MIT"
        assert response.modelfile == "FROM llama2:7b"
        assert response.parameters == "temperature 0.7"
        assert response.template == "{{ .Prompt }}"
        assert response.details.family == "llama"

    def test_show_response_default_license(self):
        """Test show response with default license."""
        # Arrange
        details = OllamaModelDetails()

        # Act
        response = OllamaShowResponse(
            modelfile="FROM test",
            parameters="temp 0.7",
            template="{{ .Prompt }}",
            details=details
        )

        # Assert
        assert response.license == "MIT"  # Default value


class TestOllamaGenerateResponse:
    """Test cases for OllamaGenerateResponse schema."""

    def test_valid_generate_response(self):
        """Test creating a valid generate response."""
        # Act
        response = OllamaGenerateResponse(
            model="llama2:7b",
            created_at="2023-12-01T10:00:00Z",
            response="Generated text",
            done=True,
            context=[1, 2, 3],
            total_duration=1000000000,
            load_duration=100000000,
            prompt_eval_count=10,
            prompt_eval_duration=50000000,
            eval_count=20,
            eval_duration=900000000
        )

        # Assert
        assert response.model == "llama2:7b"
        assert response.response == "Generated text"
        assert response.done is True
        assert response.context == [1, 2, 3]
        assert response.total_duration == 1000000000

    def test_generate_response_default_context(self):
        """Test generate response with default context."""
        # Act
        response = OllamaGenerateResponse(
            model="llama2:7b",
            created_at="2023-12-01T10:00:00Z",
            response="Generated text",
            done=True
        )

        # Assert
        assert response.context == []  # Default empty list

    def test_generate_response_optional_fields(self):
        """Test generate response with optional fields as None."""
        # Act
        response = OllamaGenerateResponse(
            model="llama2:7b",
            created_at="2023-12-01T10:00:00Z",
            response="Generated text",
            done=True,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None
        )

        # Assert
        assert response.total_duration is None
        assert response.load_duration is None
        assert response.prompt_eval_count is None


class TestOllamaChatResponse:
    """Test cases for OllamaChatResponse schema."""

    def test_valid_chat_response(self):
        """Test creating a valid chat response."""
        # Arrange
        message = OllamaMessage(role="assistant", content="Hello there!")

        # Act
        response = OllamaChatResponse(
            model="llama2:7b",
            created_at="2023-12-01T10:00:00Z",
            message=message,
            done=True
        )

        # Assert
        assert response.model == "llama2:7b"
        assert response.message.role == "assistant"
        assert response.message.content == "Hello there!"
        assert response.done is True


class TestOllamaPullResponse:
    """Test cases for OllamaPullResponse schema."""

    def test_valid_pull_response(self):
        """Test creating a valid pull response."""
        # Act
        response = OllamaPullResponse(
            status="pulling llama2:7b",
            total=1000000000,
            completed=500000000
        )

        # Assert
        assert response.status == "pulling llama2:7b"
        assert response.total == 1000000000
        assert response.completed == 500000000

    def test_pull_response_optional_fields(self):
        """Test pull response with optional fields as None."""
        # Act
        response = OllamaPullResponse(
            status="pulling llama2:7b",
            total=None,
            completed=None
        )

        # Assert
        assert response.status == "pulling llama2:7b"
        assert response.total is None
        assert response.completed is None


class TestOllamaHealthResponse:
    """Test cases for OllamaHealthResponse schema."""

    def test_valid_health_response(self):
        """Test creating a valid health response."""
        # Act
        response = OllamaHealthResponse(
            message="Custom message",
            status="healthy",
            version="1.0.0"
        )

        # Assert
        assert response.message == "Custom message"
        assert response.status == "healthy"
        assert response.version == "1.0.0"

    def test_health_response_defaults(self):
        """Test health response with default values."""
        # Act
        response = OllamaHealthResponse()

        # Assert
        assert response.message == "Ollama is running"
        assert response.status == "ok"
        assert response.version == "0.1.46"


class TestOllamaVersionResponse:
    """Test cases for OllamaVersionResponse schema."""

    def test_valid_version_response(self):
        """Test creating a valid version response."""
        # Act
        response = OllamaVersionResponse(version="1.0.0")

        # Assert
        assert response.version == "1.0.0"

    def test_version_response_default(self):
        """Test version response with default value."""
        # Act
        response = OllamaVersionResponse()

        # Assert
        assert response.version == "0.1.46"


class TestOllamaErrorResponse:
    """Test cases for OllamaErrorResponse schema."""

    def test_valid_error_response(self):
        """Test creating a valid error response."""
        # Act
        response = OllamaErrorResponse(
            detail="Something went wrong",
            error_code=500
        )

        # Assert
        assert response.detail == "Something went wrong"
        assert response.error_code == 500

    def test_error_response_without_code(self):
        """Test error response without error code."""
        # Act
        response = OllamaErrorResponse(detail="Error occurred")

        # Assert
        assert response.detail == "Error occurred"
        assert response.error_code is None

    def test_error_response_missing_detail(self):
        """Test validation error for missing detail."""
        # Act & Assert
        with pytest.raises(ValidationError):
            OllamaErrorResponse()  # Missing detail