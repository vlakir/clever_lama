"""Constants for CleverLama application."""

# HTTP Status Codes
HTTP_OK = 200
HTTP_SERVER_ERROR = 500
RESPONSE_CODE_OK = 200
HTTP_BAD_GATEWAY_ERROR = 502

# API Configuration
API_VERSION = '0.1.46'
DEFAULT_MODEL = 'clever_lama_stub'

# HTTP Headers
DEFAULT_SERVER_HEADER = 'Ollama/0.1.46'
DEFAULT_CONTENT_TYPE = 'application/json; charset=utf-8'

# Model Configuration
DEFAULT_MODEL_SIZE = 5_000_000_000

# Timing Configuration (in seconds)
HEALTH_PROBE_DELAY = 2.0

RESPONSE_PREFIX = '\U0001f999'
