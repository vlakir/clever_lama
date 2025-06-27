"""Request and response models for CleverLama."""

from typing import Annotated

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


class OllamaGenerateRequest(BaseModel):
    """Request model for Ollama generate endpoint."""

    model: str = Field(min_length=1, description='Model name')
    prompt: str = Field(min_length=1, description='Prompt for generation')
    stream: bool | None = Field(
        default=False, description='Whether to stream the response'
    )
    options: OllamaOptions | None = Field(None, description='Generation options')


class OllamaChatRequest(BaseModel):
    """Request model for Ollama chat endpoint."""

    model: str = Field(min_length=1, description='Model name')
    messages: Annotated[
        list[OllamaMessage], Field(min_length=1, description='List of messages')
    ]
    stream: bool | None = Field(
        default=False, description='Whether to stream the response'
    )
    options: OllamaOptions | None = Field(None, description='Generation options')


class OllamaShowRequest(BaseModel):
    """Request model for Ollama show endpoint."""

    name: str = Field(min_length=1, description='Model name')


class OllamaPullRequest(BaseModel):
    """Request model for Ollama pull endpoint."""

    name: str = Field(min_length=1, description='Model name')
    stream: bool | None = Field(
        default=True, description='Whether to stream the response'
    )


class OllamaModelDetails(BaseModel):
    """Model details for Ollama responses."""

    parent_model: str = Field('', description='Parent model')
    format: str = Field('gguf', description='Model format')
    family: str = Field('', description='Model family')
    families: list[str] = Field(default_factory=list, description='Model families')
    parameter_size: str = Field('Unknown', description='Parameter size')
    quantization_level: str = Field('Q4_0', description='Quantization level')


class OllamaModel(BaseModel):
    """Model information for Ollama responses."""

    name: str = Field(description='Model name')
    model: str = Field(description='Model identifier')
    modified_at: str = Field(description='Last modified timestamp')
    size: int = Field(description='Model size in bytes')
    digest: str = Field(description='Model digest')
    details: OllamaModelDetails = Field(description='Model details')


class OllamaModelsResponse(BaseModel):
    """Response model for Ollama models list."""

    models: list[OllamaModel] = Field(description='List of available models')


class OllamaShowResponse(BaseModel):
    """Response model for Ollama show endpoint."""

    license: str = Field('MIT', description='Model license')
    modelfile: str = Field(description='Model file content')
    parameters: str = Field(description='Model parameters')
    template: str = Field(description='Model template')
    details: OllamaModelDetails = Field(description='Model details')


class OllamaGenerateResponse(BaseModel):
    """Response model for Ollama generate endpoint."""

    model: str = Field(description='Model name')
    created_at: str = Field(description='Creation timestamp')
    response: str = Field(description='Generated response')
    done: bool = Field(description='Whether generation is complete')
    context: list[int] = Field(default_factory=list, description='Context tokens')
    total_duration: int | None = Field(
        None, description='Total duration in nanoseconds'
    )
    load_duration: int | None = Field(None, description='Load duration in nanoseconds')
    prompt_eval_count: int | None = Field(
        None, description='Prompt evaluation token count'
    )
    prompt_eval_duration: int | None = Field(
        None, description='Prompt evaluation duration'
    )
    eval_count: int | None = Field(None, description='Evaluation token count')
    eval_duration: int | None = Field(None, description='Evaluation duration')


class OllamaChatResponse(BaseModel):
    """Response model for Ollama chat endpoint."""

    model: str = Field(description='Model name')
    created_at: str = Field(description='Creation timestamp')
    message: OllamaMessage = Field(description='Response message')
    done: bool = Field(description='Whether generation is complete')


class OllamaPullResponse(BaseModel):
    """Response model for Ollama pull endpoint."""

    status: str = Field(description='Pull status')
    total: int | None = Field(None, description='Total size')
    completed: int | None = Field(None, description='Completed size')


class OllamaHealthResponse(BaseModel):
    """Response model for health check."""

    message: str = Field('Ollama is running', description='Health message')
    status: str = Field('ok', description='Health status')
    version: str = Field('0.1.46', description='API version')


class OllamaVersionResponse(BaseModel):
    """Response model for version endpoint."""

    version: str = Field('0.1.46', description='API version')


class OllamaErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(description='Error detail')
    error_code: str | None = Field(None, description='Error code')
