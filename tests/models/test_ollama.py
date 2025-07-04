"""Tests for Ollama models."""

import pytest
from pydantic import ValidationError

from models.ollama import (
    OllamaMessage,
    OllamaOptions,
    OllamaModelDetails,
    OllamaModel,
)


class TestOllamaMessage:
    """Test cases for OllamaMessage model."""

    def test_valid_message_creation(self):
        """Test creating a valid OllamaMessage."""

        message = OllamaMessage(role="user", content="Hello, world!")

        
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_valid_roles(self):
        """Test all valid roles for OllamaMessage."""
        valid_roles = ["system", "user", "assistant"]
        
        for role in valid_roles:
            
            message = OllamaMessage(role=role, content="Test content")
            
            
            assert message.role == role
            assert message.content == "Test content"

    def test_invalid_role_validation(self):
        """Test validation error for invalid role."""
         
        with pytest.raises(ValidationError) as exc_info:
            OllamaMessage(role="invalid_role", content="Test content")
        
        assert "Role must be one of" in str(exc_info.value)

    def test_empty_content_allowed(self):
        """Test that empty content is allowed."""
        
        message = OllamaMessage(role="user", content="")
        
        
        assert message.role == "user"
        assert message.content == ""

    def test_role_case_sensitivity(self):
        """Test that role validation is case sensitive."""
         
        with pytest.raises(ValidationError):
            OllamaMessage(role="USER", content="Test content")
        
        with pytest.raises(ValidationError):
            OllamaMessage(role="User", content="Test content")

    def test_missing_required_fields(self):
        """Test validation error when required fields are missing."""
         
        with pytest.raises(ValidationError):
            OllamaMessage(role="user")  # Missing content
        
        with pytest.raises(ValidationError):
            OllamaMessage(content="Test content")  # Missing role


class TestOllamaOptions:
    """Test cases for OllamaOptions model."""

    def test_default_options(self):
        """Test creating OllamaOptions with default values."""
        
        options = OllamaOptions()
        
        
        assert options.temperature is None
        assert options.top_p is None
        assert options.max_tokens is None
        assert options.num_predict is None
        assert options.num_ctx is None

    def test_valid_temperature_values(self):
        """Test valid temperature values."""
        valid_temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        for temp in valid_temperatures:
            
            options = OllamaOptions(temperature=temp)
            
            
            assert options.temperature == temp

    def test_invalid_temperature_values(self):
        """Test invalid temperature values."""
        invalid_temperatures = [-0.1, 2.1, -1.0, 3.0]
        
        for temp in invalid_temperatures:
             
            with pytest.raises(ValidationError):
                OllamaOptions(temperature=temp)

    def test_valid_top_p_values(self):
        """Test valid top_p values."""
        valid_top_p = [0.0, 0.1, 0.5, 0.9, 1.0]
        
        for top_p in valid_top_p:
            
            options = OllamaOptions(top_p=top_p)
            
            
            assert options.top_p == top_p

    def test_invalid_top_p_values(self):
        """Test invalid top_p values."""
        invalid_top_p = [-0.1, 1.1, -1.0, 2.0]
        
        for top_p in invalid_top_p:
             
            with pytest.raises(ValidationError):
                OllamaOptions(top_p=top_p)

    def test_valid_max_tokens_values(self):
        """Test valid max_tokens values."""
        valid_max_tokens = [1, 100, 1000, 4096]
        
        for max_tokens in valid_max_tokens:
            
            options = OllamaOptions(max_tokens=max_tokens)
            
            
            assert options.max_tokens == max_tokens

    def test_invalid_max_tokens_values(self):
        """Test invalid max_tokens values."""
        invalid_max_tokens = [0, -1, -100]
        
        for max_tokens in invalid_max_tokens:
             
            with pytest.raises(ValidationError):
                OllamaOptions(max_tokens=max_tokens)

    def test_valid_num_predict_values(self):
        """Test valid num_predict values."""
        valid_num_predict = [1, 50, 200, 1000]
        
        for num_predict in valid_num_predict:
            
            options = OllamaOptions(num_predict=num_predict)
            
            
            assert options.num_predict == num_predict

    def test_invalid_num_predict_values(self):
        """Test invalid num_predict values."""
        invalid_num_predict = [0, -1, -50]
        
        for num_predict in invalid_num_predict:
             
            with pytest.raises(ValidationError):
                OllamaOptions(num_predict=num_predict)

    def test_valid_num_ctx_values(self):
        """Test valid num_ctx values."""
        valid_num_ctx = [1, 512, 2048, 4096]
        
        for num_ctx in valid_num_ctx:
            
            options = OllamaOptions(num_ctx=num_ctx)
            
            
            assert options.num_ctx == num_ctx

    def test_invalid_num_ctx_values(self):
        """Test invalid num_ctx values."""
        invalid_num_ctx = [0, -1, -512]
        
        for num_ctx in invalid_num_ctx:
             
            with pytest.raises(ValidationError):
                OllamaOptions(num_ctx=num_ctx)

    def test_all_options_together(self):
        """Test setting all options together."""
        
        options = OllamaOptions(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            num_predict=500,
            num_ctx=2048
        )
        
        
        assert options.temperature == 0.7
        assert options.top_p == 0.9
        assert options.max_tokens == 1000
        assert options.num_predict == 500
        assert options.num_ctx == 2048


class TestOllamaModelDetails:
    """Test cases for OllamaModelDetails model."""

    def test_default_model_details(self):
        """Test creating OllamaModelDetails with default values."""
        
        details = OllamaModelDetails()
        
        
        assert details.parent_model == ""
        assert details.format == "gguf"
        assert details.family == ""
        assert details.families == []
        assert details.parameter_size == "Unknown"
        assert details.quantization_level == "Q4_0"
        assert details.num_parameters == 7_000_000_000

    def test_custom_model_details(self):
        """Test creating OllamaModelDetails with custom values."""
        
        details = OllamaModelDetails(
            parent_model="llama2",
            format="ggml",
            family="llama",
            families=["llama", "chat"],
            parameter_size="7B",
            quantization_level="Q8_0",
            num_parameters=6_700_000_000
        )
        
        
        assert details.parent_model == "llama2"
        assert details.format == "ggml"
        assert details.family == "llama"
        assert details.families == ["llama", "chat"]
        assert details.parameter_size == "7B"
        assert details.quantization_level == "Q8_0"
        assert details.num_parameters == 6_700_000_000

    def test_empty_families_list(self):
        """Test that families can be an empty list."""
        
        details = OllamaModelDetails(families=[])
        
        
        assert details.families == []

    def test_single_family_in_list(self):
        """Test families with a single item."""
        
        details = OllamaModelDetails(families=["llama"])
        
        
        assert details.families == ["llama"]


class TestOllamaModel:
    """Test cases for OllamaModel model."""

    def test_valid_model_creation(self):
        """Test creating a valid OllamaModel."""
        
        details = OllamaModelDetails(
            family="llama",
            parameter_size="7B"
        )
        
        
        model = OllamaModel(
            name="llama2:7b",
            model="llama2:7b",
            modified_at="2023-12-01T10:00:00Z",
            size=3_800_000_000,
            digest="sha256:abc123",
            details=details
        )
        
        
        assert model.name == "llama2:7b"
        assert model.model == "llama2:7b"
        assert model.modified_at == "2023-12-01T10:00:00Z"
        assert model.size == 3_800_000_000
        assert model.digest == "sha256:abc123"
        assert model.details.family == "llama"
        assert model.details.parameter_size == "7B"

    def test_model_with_default_details(self):
        """Test creating OllamaModel with default details."""
        
        details = OllamaModelDetails()
        
        
        model = OllamaModel(
            name="test-model",
            model="test-model",
            modified_at="2023-12-01T10:00:00Z",
            size=1_000_000_000,
            digest="sha256:def456",
            details=details
        )
        
        
        assert model.name == "test-model"
        assert model.details.format == "gguf"  # Default value
        assert model.details.num_parameters == 7_000_000_000  # Default value

    def test_missing_required_fields(self):
        """Test validation error when required fields are missing."""
        details = OllamaModelDetails()
        
        # Test missing name
        with pytest.raises(ValidationError):
            OllamaModel(
                model="test-model",
                modified_at="2023-12-01T10:00:00Z",
                size=1_000_000_000,
                digest="sha256:abc123",
                details=details
            )
        
        # Test missing model
        with pytest.raises(ValidationError):
            OllamaModel(
                name="test-model",
                modified_at="2023-12-01T10:00:00Z",
                size=1_000_000_000,
                digest="sha256:abc123",
                details=details
            )
        
        # Test missing details
        with pytest.raises(ValidationError):
            OllamaModel(
                name="test-model",
                model="test-model",
                modified_at="2023-12-01T10:00:00Z",
                size=1_000_000_000,
                digest="sha256:abc123"
            )

    def test_zero_size_model(self):
        """Test model with zero size."""
        
        details = OllamaModelDetails()
        
        
        model = OllamaModel(
            name="empty-model",
            model="empty-model",
            modified_at="2023-12-01T10:00:00Z",
            size=0,
            digest="sha256:empty",
            details=details
        )
        
        
        assert model.size == 0

    def test_large_size_model(self):
        """Test model with very large size."""
        
        details = OllamaModelDetails()
        large_size = 100_000_000_000  # 100GB
        
        
        model = OllamaModel(
            name="large-model",
            model="large-model",
            modified_at="2023-12-01T10:00:00Z",
            size=large_size,
            digest="sha256:large",
            details=details
        )
        
        
        assert model.size == large_size

    def test_model_serialization(self):
        """Test model serialization to dict."""
        
        details = OllamaModelDetails(family="llama", parameter_size="7B")
        model = OllamaModel(
            name="test-model",
            model="test-model",
            modified_at="2023-12-01T10:00:00Z",
            size=1_000_000_000,
            digest="sha256:test",
            details=details
        )
        
        
        model_dict = model.model_dump()
        
        
        assert model_dict["name"] == "test-model"
        assert model_dict["model"] == "test-model"
        assert model_dict["size"] == 1_000_000_000
        assert model_dict["details"]["family"] == "llama"
        assert model_dict["details"]["parameter_size"] == "7B"