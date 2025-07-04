"""Configuration management for CleverLama."""

import warnings
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_base_url: str = Field(
        default='https://api.aitunnel.ru/v1',
        description='Base URL for the external API provider',
    )
    api_key: str = Field(default='', description='API key for the external provider')

    # Server Configuration
    host: str = Field(default='0.0.0.0', description='Host to bind the server to')  # noqa: S104
    port: int = Field(default=11434, description='Port to bind the server to')

    # Logging Configuration
    log_level: str = Field(default='INFO', description='Logging level')

    # Cache Configuration
    cache_duration_minutes: int = Field(
        default=10,
        description='Cache duration in minutes',
    )

    # HTTP Client Configuration
    request_timeout: float = Field(
        default=60.0,
        description='HTTP request timeout in seconds',
    )

    # Security Configuration
    allowed_origins: list[str] = Field(
        default=['http://localhost:3000', 'http://localhost:8080'],
        description='Allowed CORS origins',
    )

    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100,
        description='Number of requests per minute per IP',
    )

    @field_validator('api_key')
    @staticmethod
    def api_key_must_not_be_empty(v: str) -> str:
        """Validate that API key is not empty."""
        if not v:
            # Allow empty API key during initialization but warn about it
            # The application will check for this at runtime
            warnings.warn(
                'API_KEY is not set. The application may not function properly.',
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator('log_level')
    @staticmethod
    def log_level_must_be_valid(v: str) -> str:
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            msg = f'LOG_LEVEL must be one of {valid_levels}'
            raise ValueError(msg)
        return v.upper()

    @field_validator('allowed_origins', mode='before')
    @staticmethod
    def parse_allowed_origins(v: str | list[str]) -> list[str]:
        """Parse allowed origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    # Calculate path to .env file in project root
    _current_file = Path(__file__)
    _project_root = _current_file.parent.parent
    _env_file = _project_root / '.env'

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding='utf-8',
        env_prefix='',
    )


# Global settings instance
settings = Settings()
