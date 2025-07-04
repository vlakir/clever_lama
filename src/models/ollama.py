"""Models for Ollama."""

from pydantic import BaseModel, Field, field_validator


class OllamaMessage(BaseModel):
    """Message model for Ollama chat requests."""

    role: str = Field(description='Role of the message sender')
    content: str = Field(description='Content of the message')

    @field_validator('role')
    @staticmethod
    def validate_role(v: str) -> str:
        """Validate message role."""
        allowed_roles = ['system', 'user', 'assistant']
        if v not in allowed_roles:
            msg = f'Role must be one of {allowed_roles}'
            raise ValueError(msg)
        return v


class OllamaOptions(BaseModel):
    """Options for Ollama requests."""

    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description='Temperature for generation'
    )
    top_p: float | None = Field(
        None, ge=0.0, le=1.0, description='Top-p for generation'
    )
    max_tokens: int | None = Field(None, ge=1, description='Maximum tokens to generate')
    num_predict: int | None = Field(
        None, ge=1, description='Number of tokens to predict'
    )
    num_ctx: int | None = Field(None, ge=1, description='Context window size')


class OllamaModelDetails(BaseModel):
    """Model details for Ollama responses."""

    parent_model: str = Field('', description='Parent model')
    format: str = Field('gguf', description='Model format')
    family: str = Field('', description='Model family')
    families: list[str] = Field(default_factory=list, description='Model families')
    parameter_size: str = Field('Unknown', description='Parameter size')
    quantization_level: str = Field('Q4_0', description='Quantization level')
    num_parameters: int = Field(7_000_000_000, description='Number of parameters')


class OllamaModel(BaseModel):
    """Model information for Ollama responses."""

    name: str = Field(description='Model name')
    model: str = Field(description='Model identifier')
    modified_at: str = Field(description='Last modified timestamp')
    size: int = Field(description='Model size in bytes')
    digest: str = Field(description='Model digest')
    details: OllamaModelDetails = Field(description='Model details')
