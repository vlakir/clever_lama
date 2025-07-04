"""Tests for Ollama API dependencies."""

from ports.api.ollama.depends import OpenAIServiceDep
from services.proxy import OpenAIService


class TestOpenAIServiceDep:
    """Test cases for OpenAIServiceDep dependency."""

    def test_openai_service_dep_type(self):
        """Test that OpenAIServiceDep is properly typed."""
        # The dependency should be an Annotated type
        assert hasattr(OpenAIServiceDep, '__origin__')
        assert hasattr(OpenAIServiceDep, '__metadata__')

    def test_openai_service_dependency_injection(self):
        """Test OpenAIService dependency injection."""
         
        service = OpenAIService()

        
        assert service is not None
        assert isinstance(service, OpenAIService)

    def test_openai_service_dep_annotation(self):
        """Test OpenAIServiceDep annotation structure."""
        # Check that it's properly annotated with Depends
        metadata = OpenAIServiceDep.__metadata__
        assert len(metadata) > 0

        # The metadata should contain a Depends instance
        depends_found = False
        for item in metadata:
            if hasattr(item, 'dependency'):
                depends_found = True
                assert item.dependency == OpenAIService
                break

        assert depends_found, "Depends annotation not found in metadata"

    def test_openai_service_creation(self):
        """Test that OpenAIService can be created successfully."""
        
        service = OpenAIService()

        
        assert service is not None
        assert isinstance(service, OpenAIService)
        assert hasattr(service, 'call_api')
        assert hasattr(service, 'get_models')
        assert hasattr(service, 'get_stream')

    def test_dependency_singleton_behavior(self):
        """Test dependency behavior (should create new instances)."""
        
        service1 = OpenAIService()
        service2 = OpenAIService()

        
        # FastAPI Depends creates new instances by default
        assert service1 != service2  # Different instances
