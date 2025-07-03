"""Tests for the main application module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError

from clever_lama.main import (
    app,
    startup_event,
    shutdown_event,
    lifespan,
    custom_key_builder,
    validation_exception_handler,
    add_ollama_headers,
)


class TestCustomKeyBuilder:
    """Test cases for custom_key_builder function."""

    def test_custom_key_builder_basic(self):
        """Test custom_key_builder with basic parameters."""
        
        def dummy_func():
            pass

        args = (1, 2, 3)
        kwargs = {"param1": "value1", "param2": "value2"}

        
        result = custom_key_builder(dummy_func, args=args, kwargs=kwargs)

        
        expected = f"{dummy_func.__module__}:{dummy_func.__name__}:{args}:{kwargs}"
        assert result == expected

    def test_custom_key_builder_with_filtered_kwargs(self):
        """Test custom_key_builder filters out specific kwargs."""
        
        def dummy_func():
            pass

        args = (1, 2)
        kwargs = {
            "param1": "value1",
            "service": "should_be_filtered",
            "request": "should_be_filtered",
            "response": "should_be_filtered",
            "args": "should_be_filtered",
            "kwargs": "should_be_filtered",
            "param2": "value2"
        }

        
        result = custom_key_builder(dummy_func, args=args, kwargs=kwargs)

        
        expected_kwargs = {"param1": "value1", "param2": "value2"}
        expected = f"{dummy_func.__module__}:{dummy_func.__name__}:{args}:{expected_kwargs}"
        assert result == expected

    def test_custom_key_builder_none_kwargs(self):
        """Test custom_key_builder with None kwargs."""
        
        def dummy_func():
            pass

        args = (1, 2)

        
        result = custom_key_builder(dummy_func, args=args, kwargs=None)

        
        expected = f"{dummy_func.__module__}:{dummy_func.__name__}:{args}:{{}}"
        assert result == expected

    def test_custom_key_builder_empty_args(self):
        """Test custom_key_builder with empty args."""
        
        def dummy_func():
            pass

        kwargs = {"param1": "value1"}

        
        result = custom_key_builder(dummy_func, args=(), kwargs=kwargs)

        
        expected = f"{dummy_func.__module__}:{dummy_func.__name__}:():{kwargs}"
        assert result == expected


class TestStartupEvent:
    """Test cases for startup_event function."""

    @pytest.mark.asyncio
    @patch('clever_lama.main.client_holder')
    @patch('clever_lama.main.FastAPICache')
    @patch('clever_lama.main.gateway')
    @patch('clever_lama.main.settings')
    @patch('clever_lama.main.asyncio.sleep')
    async def test_startup_event_success(self, mock_sleep, mock_settings, mock_gateway, mock_cache, mock_client_holder):
        """Test successful startup event."""
        
        mock_settings.api_base_url = "https://api.example.com"
        mock_settings.api_key = "test-key"
        mock_settings.request_timeout = 30
        mock_gateway.health_check_external_api = AsyncMock()

        
        await startup_event()

        
        assert mock_client_holder.client is not None
        mock_cache.init.assert_called_once()
        mock_sleep.assert_called_once()
        mock_gateway.health_check_external_api.assert_called_once()

    @pytest.mark.asyncio
    @patch('clever_lama.main.client_holder')
    @patch('clever_lama.main.FastAPICache')
    @patch('clever_lama.main.gateway')
    @patch('clever_lama.main.settings')
    @patch('clever_lama.main.shutdown_event')
    async def test_startup_event_exception(self, mock_shutdown, mock_settings, mock_gateway, mock_cache, mock_client_holder):
        """Test startup event with exception."""
        
        mock_settings.api_base_url = "https://api.example.com"
        mock_settings.api_key = "test-key"
        mock_settings.request_timeout = 30
        mock_gateway.health_check_external_api = AsyncMock(side_effect=Exception("Health check failed"))
        mock_shutdown.return_value = AsyncMock()

         
        with pytest.raises(Exception, match="Health check failed"):
            await startup_event()

        mock_shutdown.assert_called_once()


class TestShutdownEvent:
    """Test cases for shutdown_event function."""

    @pytest.mark.asyncio
    @patch('clever_lama.main.client_holder')
    async def test_shutdown_event_with_client(self, mock_client_holder):
        """Test shutdown event with existing client."""
        
        mock_client = AsyncMock()
        mock_client_holder.client = mock_client

        
        await shutdown_event()

        
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    @patch('clever_lama.main.client_holder')
    async def test_shutdown_event_without_client(self, mock_client_holder):
        """Test shutdown event without client."""
        
        mock_client_holder.client = None

        
        await shutdown_event()


class TestLifespan:
    """Test cases for lifespan context manager."""

    @pytest.mark.asyncio
    @patch('clever_lama.main.startup_event')
    @patch('clever_lama.main.shutdown_event')
    async def test_lifespan_success(self, mock_shutdown, mock_startup):
        """Test successful lifespan context manager."""
        
        mock_startup.return_value = AsyncMock()
        mock_shutdown.return_value = AsyncMock()
        mock_app = MagicMock()

        
        async with lifespan(mock_app):
            pass

        
        mock_startup.assert_called_once()
        mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    @patch('clever_lama.main.startup_event')
    @patch('clever_lama.main.shutdown_event')
    async def test_lifespan_startup_exception(self, mock_shutdown, mock_startup):
        """Test lifespan with startup exception."""
        
        mock_startup.side_effect = Exception("Startup failed")
        mock_app = MagicMock()

         
        with pytest.raises(Exception, match="Startup failed"):
            async with lifespan(mock_app):
                pass

        mock_startup.assert_called_once()
        # shutdown_event should NOT be called from lifespan when startup fails
        # because the exception prevents reaching the yield point
        mock_shutdown.assert_not_called()


class TestValidationExceptionHandler:
    """Test cases for validation_exception_handler function."""

    @pytest.mark.asyncio
    async def test_validation_exception_handler(self):
        """Test validation exception handler."""
        
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        mock_exc = RequestValidationError([
            {"type": "missing", "loc": ["field1"], "msg": "field required"}
        ])

        
        response = await validation_exception_handler(mock_request, mock_exc)

        
        assert response.status_code == 422
        response_body = response.body.decode()
        assert "Data validation error" in response_body
        assert "errors" in response_body


class TestAddOllamaHeaders:
    """Test cases for add_ollama_headers middleware."""

    @pytest.mark.asyncio
    async def test_add_ollama_headers(self):
        """Test add_ollama_headers middleware."""
        
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.headers = {}

        async def mock_call_next(request):
            return mock_response

        
        result = await add_ollama_headers(mock_request, mock_call_next)

        
        assert result == mock_response
        assert 'server' in result.headers
        assert 'content-type' in result.headers
        assert 'x-process-time' in result.headers


class TestFastAPIApp:
    """Test cases for FastAPI application configuration."""

    def test_app_configuration(self):
        """Test FastAPI app configuration."""
        
        assert isinstance(app, FastAPI)
        assert app.title == "CleverLama"
        assert app.version == "1.0.0"
        assert "Bridge between Ollama API and OpenAI-compatible providers" in app.description
        assert app.docs_url is None
        assert app.redoc_url is None

    def test_app_has_middleware(self):
        """Test that app has required middleware."""
        # Get middleware stack
        middleware_stack = app.user_middleware

        # Check that CORS middleware is present
        cors_middleware_found = any(
            middleware.cls.__name__ == "CORSMiddleware" 
            for middleware in middleware_stack
        )
        assert cors_middleware_found

    def test_app_has_routers(self):
        """Test that app has required routers."""
        # Get all routes
        routes = app.routes
        route_paths = [route.path for route in routes if hasattr(route, 'path')]

        # Check that expected routes are present
        assert "/" in route_paths  # Root router
        # API routes should be present (they have /api prefix)
        api_routes = [path for path in route_paths if path.startswith("/api")]
        assert len(api_routes) > 0

    def test_app_exception_handlers(self):
        """Test that app has exception handlers."""
        # Check that RequestValidationError handler is registered
        assert RequestValidationError in app.exception_handlers


class TestAppIntegration:
    """Integration tests for the FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @patch('clever_lama.main.startup_event')
    @patch('clever_lama.main.shutdown_event')
    def test_health_endpoint(self, mock_shutdown, mock_startup, client):
        """Test health check endpoint."""
        
        mock_startup.return_value = AsyncMock()
        mock_shutdown.return_value = AsyncMock()

        
        response = client.get("/")

        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "Ollama is running" in data["message"]

    @patch('clever_lama.main.startup_event')
    @patch('clever_lama.main.shutdown_event')
    def test_version_endpoint(self, mock_shutdown, mock_startup, client):
        """Test version endpoint."""
        
        mock_startup.return_value = AsyncMock()
        mock_shutdown.return_value = AsyncMock()

        
        response = client.get("/api/version")

        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data

    @patch('clever_lama.main.startup_event')
    @patch('clever_lama.main.shutdown_event')
    def test_cors_headers(self, mock_shutdown, mock_startup, client):
        """Test CORS headers are present."""
        
        mock_startup.return_value = AsyncMock()
        mock_shutdown.return_value = AsyncMock()

        
        response = client.options("/", headers={"Origin": "http://localhost:3000"})

        
        assert "access-control-allow-origin" in response.headers

    @patch('clever_lama.main.startup_event')
    @patch('clever_lama.main.shutdown_event')
    def test_ollama_headers_middleware(self, mock_shutdown, mock_startup, client):
        """Test Ollama headers middleware."""
        
        mock_startup.return_value = AsyncMock()
        mock_shutdown.return_value = AsyncMock()

        
        response = client.get("/")

        
        assert "server" in response.headers
        assert "content-type" in response.headers
        assert "x-process-time" in response.headers

    @patch('clever_lama.main.startup_event')
    @patch('clever_lama.main.shutdown_event')
    def test_validation_error_handling(self, mock_shutdown, mock_startup, client):
        """Test validation error handling."""
        
        mock_startup.return_value = AsyncMock()
        mock_shutdown.return_value = AsyncMock()

        response = client.post("/api/generate", json={})  # Missing required fields

        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "errors" in data
