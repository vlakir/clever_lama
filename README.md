# CleverLama

Bridge between Ollama API and OpenAI-compatible providers

## Description

CleverLama is an API bridge that provides an Ollama API interface for working with OpenAI-compatible providers. This allows you to use applications that expect Ollama API (e.g., JetBrains AI Assistant) with any OpenAI-compatible services (e.g., via aitunnel.ru API).

**How it works**: CleverLama accepts requests in Ollama API format, transforms them to OpenAI API format, sends them to an external provider, receives the response, and transforms it back to Ollama format.

## Architecture

The project is built on FastAPI and uses the Ports & Adapters architectural pattern:
- **Ports/API**: Ollama-compatible HTTP endpoints
- **Ports/SPI**: Integration with OpenAI-compatible providers  
- **Services**: Business logic for request transformation
- **Models**: Data models for both API formats

## Features

- ✅ Full compatibility with Ollama API
- ✅ Support for OpenAI-compatible providers
- ✅ Request transformation between formats
- ✅ Streaming response support
- ✅ Model list caching
- ✅ Docker containerization
- ✅ Structured logging
- ✅ Health checks and monitoring

## Quick Start

### Using Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/vlakir/clever-lama.git
cd clever-lama
```

2. Create a `.env` file with your settings:
```bash
API_KEY=your_api_key_here
API_BASE_URL=https://api.aitunnel.ru/v1
LOG_LEVEL=INFO
```

3. Start the service:
```bash
docker-compose up -d
```

### Local Development

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Start the server:
```bash
poetry run python src/clever_lama/main.py
```

## Usage

After startup, the service will be available at `http://localhost:11434` and provides the following endpoints:

- `GET /` - service health check
- `GET /api/version` - API version
- `GET /api/tags` - list of available models
- `POST /api/show` - model information
- `POST /api/pull` - model download
- `POST /api/generate` - text generation
- `POST /api/chat` - chat with model

## Configuration

Settings are configured via environment variables:

- `API_KEY` - API key for the provider (required)
- `API_BASE_URL` - base URL of the API provider (default: https://api.aitunnel.ru/v1)
- `LOG_LEVEL` - logging level (default: INFO)
- `HOST` - host to bind to (default: 0.0.0.0)
- `PORT` - port to bind to (default: 11434)
- `CACHE_DURATION_MINUTES` - model list cache duration in minutes (default: 10)

## Usage Examples

### Getting list of models
```bash
curl http://localhost:11434/api/tags
```

### Text generation
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt": "Tell me a joke about programmers",
    "stream": false
  }'
```

### Chat with model
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "stream": false
  }'
```

### Streaming chat
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Write a short story"}
    ],
    "stream": true
  }'
```


## Development

### Requirements

- Python 3.11-3.13
- Poetry 1.8+
- Docker (optional)

### Main Dependencies

- **FastAPI** - web framework for API
- **httpx** - HTTP client for external API requests
- **uvicorn** - ASGI server
- **pydantic-settings** - configuration management
- **python-dotenv** - environment variables loading
- **fastapi-cache2** - response caching

### Project Status

🚧 **Beta version** - project is under active development. API may change.

### Development Installation

```bash
poetry install --with dev
```

### Running Tests

```bash
poetry run pytest
```

### Linting and Formatting

```bash
./check.sh
```

## Troubleshooting

### Connection Issues

**Error**: `HTTP client not initialized`
- Check the correctness of environment variables `API_KEY` and `API_BASE_URL`
- Make sure the external API is accessible

**Error**: `502 Bad Gateway`
- Check API key validity
- Make sure the model exists at the provider
- Check application logs for detailed information

### Performance Issues

**Slow responses**:
- Increase cache duration via `CACHE_DURATION_MINUTES`
- Check network connection to the provider
- Consider using faster models

### Logging

For detailed logging, set:
```bash
LOG_LEVEL=DEBUG
```

Logs contain information about:
- Incoming requests
- Data transformation
- External API requests
- Errors and exceptions

## Compatibility

### Tested with:
- **JetBrains AI Assistant** ✅

### Supported providers:
- OpenAI API
- Azure OpenAI
- Any OpenAI-compatible APIs (aitunnel.ru, etc.)

## License

MIT License
