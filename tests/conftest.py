"""Shared test configuration and fixtures."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import os

# Корректно добавляем путь src в начало sys.path
src_path = Path(__file__).parent.parent / "src"
src_path_str = str(src_path)
if src_path_str in sys.path:
    sys.path.remove(src_path_str)
sys.path.insert(0, src_path_str)

print("Current working directory:", os.getcwd())
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', sys.path)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.api_base_url = "https://api.example.com"
    settings.api_key = "test-api-key"
    settings.request_timeout = 30
    settings.host = "localhost"
    settings.port = 8000
    settings.allowed_origins = ["*"]
    return settings


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient for testing."""
    client = AsyncMock()
    client.base_url = "https://api.example.com"
    return client


@pytest.fixture
def sample_openai_response():
    """Sample OpenAI API response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                }
            }
        ]
    }


@pytest.fixture
def sample_openai_stream_chunks():
    """Sample OpenAI streaming response chunks for testing."""
    return [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
        {"choices": [{"delta": {}}]}  # End chunk
    ]


@pytest.fixture
def sample_models_response():
    """Sample models API response for testing."""
    return {
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai"
            }
        ]
    }


@pytest.fixture
def sample_ollama_message():
    """Sample OllamaMessage for testing."""
    from models.ollama import OllamaMessage
    return OllamaMessage(role="user", content="Hello, world!")


@pytest.fixture
def sample_ollama_model():
    """Sample OllamaModel for testing."""
    from models.ollama import OllamaModel, OllamaModelDetails

    details = OllamaModelDetails(
        family="llama",
        parameter_size="7B",
        quantization_level="Q4_0"
    )

    return OllamaModel(
        name="llama2:7b",
        model="llama2:7b",
        modified_at="2023-12-01T10:00:00Z",
        size=3_800_000_000,
        digest="sha256:abc123def456",
        details=details
    )


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing."""
    from ports.api.ollama.schemas import OllamaChatRequest
    from models.ollama import OllamaMessage

    messages = [
        OllamaMessage(role="user", content="Hello"),
        OllamaMessage(role="assistant", content="Hi there!")
    ]

    return OllamaChatRequest(
        model="llama2:7b",
        messages=messages,
        stream=False
    )


@pytest.fixture
def sample_generate_request():
    """Sample generate request for testing."""
    from ports.api.ollama.schemas import OllamaGenerateRequest

    return OllamaGenerateRequest(
        model="llama2:7b",
        prompt="Tell me a joke",
        stream=False
    )


@pytest.fixture
def reset_client_holder():
    """Reset client holder before each test."""
    try:
        from ports.spi.openai.gateway import client_holder
        original_client = client_holder.client
        yield
        client_holder.client = original_client
    except ImportError:
        # Skip if module can't be imported
        yield


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = MagicMock()
    return logger


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )

# Async test configuration
# pytest_plugins = ["pytest_asyncio"]  # Commented out - install pytest-asyncio if needed
