"""Tests for Ollama API schemas."""

import pytest
from pydantic import ValidationError

from ports.api.ollama.schemas import (
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
from models.ollama import OllamaMessage, OllamaModel, OllamaModelDetails, OllamaOptions


class TestOllamaGenerateRequest:
    """Test cases for OllamaGenerateRequest schema."""

    def test_valid_generate_request(self):
        """Test creating a valid generate request."""
        
        request = OllamaGenerateRequest(
            model="llama2:7b",
            prompt="Hello, world!",
            stream=False
        )

        
        assert request.model == "llama2:7b"
        assert request.prompt == "Hello, world!"
        assert request.stream is False
        assert request.options is None

    def test_generate_request_with_options(self):
        """Test generate request with options."""
        
        options = OllamaOptions(temperature=0.7, max_tokens=100)

        
        request = OllamaGenerateRequest(
            model="llama2:7b",
            prompt="Hello, world!",
            stream=True,
            options=options
        )

        
        assert request.model == "llama2:7b"
        assert request.prompt == "Hello, world!"
        assert request.stream is True
        assert request.options.temperature == 0.7
        assert request.options.max_tokens == 100

    def test_generate_request_default_stream(self):
        """Test generate request with default stream value."""
        
        request = OllamaGenerateRequest(
            model="llama2:7b",
            prompt="Hello, world!"
        )

        
        assert request.stream is False

    def test_generate_request_empty_model(self):
        """Test validation error for empty model."""
         
        with pytest.raises(ValidationError):
            OllamaGenerateRequest(model="", prompt="Hello, world!")

    def test_generate_request_empty_prompt(self):
        """Test validation error for empty prompt."""
         
        with pytest.raises(ValidationError):
            OllamaGenerateRequest(model="llama2:7b", prompt="")

    def test_generate_request_missing_required_fields(self):
        """Test validation error for missing required fields."""
         
        with pytest.raises(ValidationError):
            OllamaGenerateRequest(model="llama2:7b")  # Missing prompt

        with pytest.raises(ValidationError):
            OllamaGenerateRequest(prompt="Hello, world!")  # Missing model


class TestOllamaChatRequest:
    """Test cases for OllamaChatRequest schema."""

    def test_valid_chat_request(self):
        """Test creating a valid chat request."""
        
        messages = [
            OllamaMessage(role="user", content="Hello"),
            OllamaMessage(role="assistant", content="Hi there!")
        ]

        
        request = OllamaChatRequest(
            model="llama2:7b",
            messages=messages,
            stream=False
        )

        
        assert request.model == "llama2:7b"
        assert len(request.messages) == 2
        assert request.messages[0].role == "user"
        assert request.messages[1].role == "assistant"
        assert request.stream is False

    def test_chat_request_with_options(self):
        """Test chat request with options."""
        
        messages = [OllamaMessage(role="user", content="Hello")]
        options = OllamaOptions(temperature=0.8, top_p=0.9)

        
        request = OllamaChatRequest(
            model="llama2:7b",
            messages=messages,
            stream=True,
            options=options
        )

        
        assert request.options.temperature == 0.8
        assert request.options.top_p == 0.9

    def test_chat_request_default_stream(self):
        """Test chat request with default stream value."""
        
        messages = [OllamaMessage(role="user", content="Hello")]

        
        request = OllamaChatRequest(
            model="llama2:7b",
            messages=messages
        )

        
        assert request.stream is False

    def test_chat_request_empty_messages(self):
        """Test validation error for empty messages list."""
         
        with pytest.raises(ValidationError):
            OllamaChatRequest(model="llama2:7b", messages=[])

    def test_chat_request_empty_model(self):
        """Test validation error for empty model."""
        
        messages = [OllamaMessage(role="user", content="Hello")]

         
        with pytest.raises(ValidationError):
            OllamaChatRequest(model="", messages=messages)

    def test_chat_request_missing_required_fields(self):
        """Test validation error for missing required fields."""
        messages = [OllamaMessage(role="user", content="Hello")]

         
        with pytest.raises(ValidationError):
            OllamaChatRequest(model="llama2:7b")  # Missing messages

        with pytest.raises(ValidationError):
            OllamaChatRequest(messages=messages)  # Missing model


class TestOllamaShowRequest:
    """Test cases for OllamaShowRequest schema."""

    def test_valid_show_request(self):
        """Test creating a valid show request."""
        
        request = OllamaShowRequest(name="llama2:7b")

        
        assert request.name == "llama2:7b"

    def test_show_request_empty_name(self):
        """Test validation error for empty name."""
         
        with pytest.raises(ValidationError):
            OllamaShowRequest(name="")

    def test_show_request_missing_name(self):
        """Test validation error for missing name."""
         
        with pytest.raises(ValidationError):
            OllamaShowRequest()


class TestOllamaPullRequest:
    """Test cases for OllamaPullRequest schema."""

    def test_valid_pull_request(self):
        """Test creating a valid pull request."""
        
        request = OllamaPullRequest(name="llama2:7b", stream=False)

        
        assert request.name == "llama2:7b"
        assert request.stream is False

    def test_pull_request_default_stream(self):
        """Test pull request with default stream value."""
        
        request = OllamaPullRequest(name="llama2:7b")

        
        assert request.stream is True  # Default is True for pull

    def test_pull_request_empty_name(self):
        """Test validation error for empty name."""
         
        with pytest.raises(ValidationError):
            OllamaPullRequest(name="")

    def test_pull_request_missing_name(self):
        """Test validation error for missing name."""
         
        with pytest.raises(ValidationError):
            OllamaPullRequest()


class TestOllamaModelsResponse:
    """Test cases for OllamaModelsResponse schema."""

    def test_valid_models_response(self):
        """Test creating a valid models response."""
        
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

        
        response = OllamaModelsResponse(models=models)

        
        assert len(response.models) == 1
        assert response.models[0].name == "llama2:7b"

    def test_empty_models_response(self):
        """Test models response with empty list."""
        
        response = OllamaModelsResponse(models=[])

        
        assert len(response.models) == 0


class TestOllamaShowResponse:
    """Test cases for OllamaShowResponse schema."""

    def test_valid_show_response(self):
        """Test creating a valid show response."""
        
        details = OllamaModelDetails(family="llama", parameter_size="7B")

        
        response = OllamaShowResponse(
            license="MIT",
            modelfile="FROM llama2:7b",
            parameters="temperature 0.7",
            template="{{ .Prompt }}",
            details=details
        )

        
        assert response.license == "MIT"
        assert response.modelfile == "FROM llama2:7b"
        assert response.parameters == "temperature 0.7"
        assert response.template == "{{ .Prompt }}"
        assert response.details.family == "llama"

    def test_show_response_default_license(self):
        """Test show response with default license."""
        
        details = OllamaModelDetails()

        
        response = OllamaShowResponse(
            modelfile="FROM test",
            parameters="temp 0.7",
            template="{{ .Prompt }}",
            details=details
        )

        
        assert response.license == "MIT"  # Default value


class TestOllamaGenerateResponse:
    """Test cases for OllamaGenerateResponse schema."""

    def test_valid_generate_response(self):
        """Test creating a valid generate response."""
        
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

        
        assert response.model == "llama2:7b"
        assert response.response == "Generated text"
        assert response.done is True
        assert response.context == [1, 2, 3]
        assert response.total_duration == 1000000000

    def test_generate_response_default_context(self):
        """Test generate response with default context."""
        
        response = OllamaGenerateResponse(
            model="llama2:7b",
            created_at="2023-12-01T10:00:00Z",
            response="Generated text",
            done=True
        )

        
        assert response.context == []  # Default empty list

    def test_generate_response_optional_fields(self):
        """Test generate response with optional fields as None."""
        
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

        
        assert response.total_duration is None
        assert response.load_duration is None
        assert response.prompt_eval_count is None


class TestOllamaChatResponse:
    """Test cases for OllamaChatResponse schema."""

    def test_valid_chat_response(self):
        """Test creating a valid chat response."""
        
        message = OllamaMessage(role="assistant", content="Hello there!")

        
        response = OllamaChatResponse(
            model="llama2:7b",
            created_at="2023-12-01T10:00:00Z",
            message=message,
            done=True
        )

        
        assert response.model == "llama2:7b"
        assert response.message.role == "assistant"
        assert response.message.content == "Hello there!"
        assert response.done is True


class TestOllamaPullResponse:
    """Test cases for OllamaPullResponse schema."""

    def test_valid_pull_response(self):
        """Test creating a valid pull response."""
        
        response = OllamaPullResponse(
            status="pulling llama2:7b",
            total=1000000000,
            completed=500000000
        )

        
        assert response.status == "pulling llama2:7b"
        assert response.total == 1000000000
        assert response.completed == 500000000

    def test_pull_response_optional_fields(self):
        """Test pull response with optional fields as None."""
        
        response = OllamaPullResponse(
            status="pulling llama2:7b",
            total=None,
            completed=None
        )

        
        assert response.status == "pulling llama2:7b"
        assert response.total is None
        assert response.completed is None


class TestOllamaHealthResponse:
    """Test cases for OllamaHealthResponse schema."""

    def test_valid_health_response(self):
        """Test creating a valid health response."""
        
        response = OllamaHealthResponse(
            message="Custom message",
            status="healthy",
            version="1.0.0"
        )

        
        assert response.message == "Custom message"
        assert response.status == "healthy"
        assert response.version == "1.0.0"

    def test_health_response_defaults(self):
        """Test health response with default values."""
        
        response = OllamaHealthResponse()

        
        assert response.message == "Ollama is running"
        assert response.status == "ok"
        assert response.version == "0.1.46"


class TestOllamaVersionResponse:
    """Test cases for OllamaVersionResponse schema."""

    def test_valid_version_response(self):
        """Test creating a valid version response."""
        
        response = OllamaVersionResponse(version="1.0.0")

        
        assert response.version == "1.0.0"

    def test_version_response_default(self):
        """Test version response with default value."""
        
        response = OllamaVersionResponse()

        
        assert response.version == "0.1.46"


class TestOllamaErrorResponse:
    """Test cases for OllamaErrorResponse schema."""

    def test_valid_error_response(self):
        """Test creating a valid error response."""
        
        response = OllamaErrorResponse(
            detail="Something went wrong",
            error_code=500
        )

        
        assert response.detail == "Something went wrong"
        assert response.error_code == 500

    def test_error_response_without_code(self):
        """Test error response without error code."""
        
        response = OllamaErrorResponse(detail="Error occurred")

        
        assert response.detail == "Error occurred"
        assert response.error_code is None

    def test_error_response_missing_detail(self):
        """Test validation error for missing detail."""
         
        with pytest.raises(ValidationError):
            OllamaErrorResponse()  # Missing detail