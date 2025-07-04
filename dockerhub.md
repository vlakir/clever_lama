# CleverLama

Bridge between Ollama API and OpenAI-compatible providers

## Description

CleverLama is an API bridge that provides an Ollama API interface for working with OpenAI-compatible providers. 
This allows you to use applications that expect Ollama API (e.g., JetBrains AI Assistant) 
with any OpenAI-compatible services (e.g., via aitunnel.ru API).

**How it works**: CleverLama accepts requests in Ollama API format, transforms them to OpenAI API format, 
sends them to an external provider, receives the response, and transforms it back to Ollama format.

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

### Using DockerHub

Run service:

```bash  
docker run -d \
  --name clever-lama \
  -p 11434:11434 \
  -e API_BASE_URL="<your_provider_api_root_path>" \
  -e API_KEY="<your_provider_api_key>" \
  -e LOG_LEVEL="INFO" \
  -e HOST=0.0.0.0 \
  -e PORT=11434 \
  --restart unless-stopped \
  --memory=512M \
  --cpus=0.5 \
  --log-driver json-file \
  --log-opt max-size=100m \
  --log-opt max-file=3 \
  --health-cmd="curl -f http://localhost:11434/" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=10s \
  clever_lama:latest  
```

*Note*: Bearer authorization is supported only. 

Stop service:

```bash
docker stop clever-lama
docker rm clever-lama
```

## Usage

After startup, the service will be available at `http://localhost:11434` and provides 
the following endpoints:

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

### Project Status

🚧 **Beta version** - project is under active development. API may change.

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


## Compatibility

### Tested with:
- **JetBrains AI Assistant** ✅

### Supported providers:
- aitunnel.ru ✅
- probably any OpenAI-compatible APIs with Bearer Authorization

## License

MIT License