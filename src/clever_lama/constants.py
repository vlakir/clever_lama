"""Constants for CleverLama application."""

# HTTP Status Codes
HTTP_OK = 200
HTTP_SERVER_ERROR = 500
RESPONSE_CODE_OK = 200

# API Configuration
API_VERSION = '0.1.46'
DEFAULT_MODEL = 'gpt-3.5-turbo'

# HTTP Headers
DEFAULT_SERVER_HEADER = 'Ollama/0.1.46'
DEFAULT_CONTENT_TYPE = 'application/json; charset=utf-8'

# Model Configuration
DEFAULT_MODEL_SIZE = 5_000_000_000

# Timing Configuration (in seconds)
HEALTH_PROBE_DELAY = 2.0

# Error Messages
SERVER_ERROR_MESSAGE = 'Внутренняя ошибка сервера'

# Cache Constants
CACHE_MODELS_KEY = 'models_cache'
CACHE_MODELS_TIMESTAMP_KEY = 'models_cache_timestamp'
