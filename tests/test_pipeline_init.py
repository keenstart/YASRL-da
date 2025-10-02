"""
Comprehensive unit tests for RAGPipeline initialization and lifecycle management.
Tests configuration validation, component initialization, error handling, and resource cleanup.
"""
import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock

import pytest

from yasrl.exceptions import ConfigurationError
from yasrl.pipeline import RAGPipeline, QueryProcessor


# Fixtures for environment setup
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set mock environment variables for testing."""
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    

@pytest.fixture
def mock_env_vars_azure(monkeypatch):
    """Fixture for Azure-specific environment variables."""
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.azure.com")


@pytest.fixture
def mock_env_vars_with_uri(monkeypatch):
    """Fixture with YASRL_POSTGRES_URI instead of individual DB params."""
    monkeypatch.setenv("YASRL_POSTGRES_URI", "postgresql://testuser:testpass@localhost:5432/testdb")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


# Test successful initialization
@pytest.mark.asyncio
async def test_pipeline_initialization_success(mock_env_vars):
    """Tests successful initialization of the RAGPipeline with all components."""
    with patch("yasrl.pipeline.ConfigurationManager") as mock_config_manager, \
         patch("yasrl.pipeline.LLMProviderFactory") as mock_llm_factory, \
         patch("yasrl.pipeline.EmbeddingProviderFactory") as mock_embedding_factory, \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager, \
         patch("yasrl.pipeline.TextProcessor") as mock_text_processor:
        
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.validate_config = MagicMock()
        mock_config_instance.get_database_config.return_value = {
            "user": "testuser",
            "password": "testpass",
            "host": "localhost",
            "port": "5432",
            "dbname": "testdb",
            "postgres_uri": "postgresql://testuser:testpass@localhost:5432/testdb",
            "table_prefix": "yasrl",
            "vector_dimensions": 1536
        }
        mock_config_instance.load_config.return_value = MagicMock(
            embedding=MagicMock(chunk_size=1024),
            chunk_overlap=200,
            retrieval_top_k=10,
            rerank_top_k=5
        )
        mock_config_instance.get_config_sources.return_value = ["Environment variables", "Default values"]
        mock_config_manager.return_value = mock_config_instance
        
        mock_llm_provider = MagicMock()
        mock_llm_factory.create.return_value = mock_llm_provider
        
        mock_embedding_provider = MagicMock()
        mock_embedding_factory.create.return_value = mock_embedding_provider
        
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.initialize = AsyncMock()
        mock_vector_store_manager.return_value = mock_vector_store_instance
        
        mock_text_processor_instance = MagicMock()
        mock_text_processor.return_value = mock_text_processor_instance
        
        # Act
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        await pipeline.initialize()
        
        # Assert
        assert pipeline.is_initialized
        assert pipeline.llm_provider == mock_llm_provider
        assert pipeline.embedding_provider == mock_embedding_provider
        assert pipeline.vector_store_manager == mock_vector_store_instance
        assert pipeline.text_processor == mock_text_processor_instance
        assert pipeline.query_processor is not None
        assert isinstance(pipeline.query_processor, QueryProcessor)
        
        # Verify calls
        mock_config_manager.assert_called_once_with(config_file=None)
        mock_config_instance.validate_config.assert_called_once_with("openai", "openai")
        mock_llm_factory.create.assert_called_once_with("openai")
        mock_embedding_factory.create.assert_called_once_with("openai")
        mock_vector_store_instance.initialize.assert_called_once()
        mock_text_processor.assert_called_once_with(chunk_size=1024, chunk_overlap=200)


@pytest.mark.asyncio
async def test_pipeline_initialization_with_config_file(mock_env_vars):
    """Tests initialization with a custom configuration file."""
    with patch("yasrl.pipeline.ConfigurationManager") as mock_config_manager, \
         patch("yasrl.pipeline.LLMProviderFactory") as mock_llm_factory, \
         patch("yasrl.pipeline.EmbeddingProviderFactory") as mock_embedding_factory, \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager:
        
        # Setup minimal mocks
        mock_config_instance = MagicMock()
        mock_config_instance.validate_config = MagicMock()
        mock_config_instance.get_database_config.return_value = {
            "postgres_uri": "postgresql://test:test@localhost/test",
            "vector_dimensions": 1536,
            "table_prefix": "custom"
        }
        mock_config_instance.load_config.return_value = MagicMock(
            embedding=MagicMock(chunk_size=512),
            chunk_overlap=100,
            retrieval_top_k=20,
            rerank_top_k=10
        )
        mock_config_instance.get_config_sources.return_value = ["Config file: custom.yaml"]
        mock_config_manager.return_value = mock_config_instance
        
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.initialize = AsyncMock()
        mock_vector_store_manager.return_value = mock_vector_store_instance
        
        # Act
        pipeline = RAGPipeline(llm="openai", embed_model="openai", config_file="custom.yaml")
        await pipeline.initialize()
        
        # Assert
        mock_config_manager.assert_called_once_with(config_file="custom.yaml")
        assert pipeline.config_file == "custom.yaml"


@pytest.mark.asyncio
async def test_pipeline_initialization_already_initialized(mock_env_vars):
    """Tests that re-initialization is skipped if pipeline is already initialized."""
    with patch("yasrl.pipeline.ConfigurationManager") as mock_config_manager, \
         patch("yasrl.pipeline.LLMProviderFactory"), \
         patch("yasrl.pipeline.EmbeddingProviderFactory"), \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager:
        
        # Setup minimal mocks
        mock_config_instance = MagicMock()
        mock_config_instance.get_database_config.return_value = {"postgres_uri": "postgresql://test:test@localhost/test"}
        mock_config_instance.load_config.return_value = MagicMock(
            embedding=MagicMock(chunk_size=1024),
            chunk_overlap=200,
            retrieval_top_k=10,
            rerank_top_k=5
        )
        mock_config_instance.get_config_sources.return_value = []
        mock_config_manager.return_value = mock_config_instance
        
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.initialize = AsyncMock()
        mock_vector_store_manager.return_value = mock_vector_store_instance
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        
        # First initialization
        await pipeline.initialize()
        mock_config_manager.reset_mock()
        
        # Second initialization should be skipped
        await pipeline.initialize()
        mock_config_manager.assert_not_called()


# Test configuration validation errors
@pytest.mark.asyncio
async def test_pipeline_initialization_missing_openai_key():
    """Tests that ConfigurationError is raised when OpenAI API key is missing."""
    pipeline = RAGPipeline(llm="openai", embed_model="openai")
    
    with pytest.raises(ConfigurationError) as exc_info:
        await pipeline.initialize()
    
    assert "OPENAI_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
async def test_pipeline_initialization_missing_azure_config():
    """Tests that ConfigurationError is raised when Azure configuration is incomplete."""
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "fake-key"}, clear=True):
        pipeline = RAGPipeline(llm="azure", embed_model="azure")
        
        with pytest.raises(ConfigurationError) as exc_info:
            await pipeline.initialize()
        
        assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)


@pytest.mark.asyncio
async def test_pipeline_initialization_missing_database_config():
    """Tests that ConfigurationError is raised when database configuration is missing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        
        with pytest.raises(ConfigurationError) as exc_info:
            await pipeline.initialize()
        
        assert "database" in str(exc_info.value).lower() or "POSTGRES" in str(exc_info.value)


@pytest.mark.asyncio
async def test_pipeline_initialization_database_connection_error(mock_env_vars):
    """Tests handling of database connection failures."""
    with patch("yasrl.pipeline.ConfigurationManager") as mock_config_manager, \
         patch("yasrl.pipeline.LLMProviderFactory"), \
         patch("yasrl.pipeline.EmbeddingProviderFactory"), \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager:
        
        mock_config_instance = MagicMock()
        mock_config_instance.get_database_config.return_value = {"postgres_uri": "postgresql://test:test@localhost/test"}
        mock_config_instance.load_config.return_value = MagicMock(embedding=MagicMock(chunk_size=1024))
        mock_config_manager.return_value = mock_config_instance
        
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.setup_schema.side_effect = Exception("Connection refused")
        mock_vector_store_manager.return_value = mock_vector_store_instance
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        
        with pytest.raises(ConfigurationError) as exc_info:
            await pipeline.initialize()
        
        assert "Failed to connect to database" in str(exc_info.value)
        assert "Connection refused" in str(exc_info.value)


# Test invalid provider names
def test_pipeline_invalid_llm_provider():
    """Tests that ValueError is raised for invalid LLM provider name."""
    with pytest.raises(ValueError) as exc_info:
        RAGPipeline(llm="invalid_llm", embed_model="openai")
    
    assert "Invalid LLM provider" in str(exc_info.value)
    assert "invalid_llm" in str(exc_info.value)


def test_pipeline_invalid_embedding_provider():
    """Tests that ValueError is raised for invalid embedding provider name."""
    with pytest.raises(ValueError) as exc_info:
        RAGPipeline(llm="openai", embed_model="invalid_embed")
    
    assert "Invalid embedding provider" in str(exc_info.value)
    assert "invalid_embed" in str(exc_info.value)


# Test context manager functionality
@pytest.mark.asyncio
async def test_pipeline_context_manager(mock_env_vars):
    """Tests the async context manager functionality."""
    with patch("yasrl.pipeline.RAGPipeline.initialize") as mock_initialize, \
         patch("yasrl.pipeline.RAGPipeline.cleanup") as mock_cleanup:
        
        mock_initialize.return_value = None
        mock_cleanup.return_value = None
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        
        async with pipeline as p:
            assert p is pipeline
            mock_initialize.assert_called_once()
            mock_cleanup.assert_not_called()
        
        mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_managed_pipeline_context(mock_env_vars):
    """Tests the managed_pipeline context manager."""
    with patch("yasrl.pipeline.RAGPipeline.initialize") as mock_initialize, \
         patch("yasrl.pipeline.RAGPipeline.cleanup") as mock_cleanup:
        
        mock_initialize.return_value = None
        mock_cleanup.return_value = None
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        
        async with pipeline.managed_pipeline() as p:
            assert p is pipeline
            mock_initialize.assert_called_once()
            mock_cleanup.assert_not_called()
        
        mock_cleanup.assert_called_once()


# Test cleanup functionality
@pytest.mark.asyncio
async def test_pipeline_cleanup(mock_env_vars):
    """Tests proper cleanup of resources."""
    with patch("yasrl.pipeline.ConfigurationManager"), \
         patch("yasrl.pipeline.LLMProviderFactory"), \
         patch("yasrl.pipeline.EmbeddingProviderFactory"), \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager:
        
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.initialize = AsyncMock()
        mock_vector_store_instance.close = AsyncMock()
        mock_vector_store_manager.return_value = mock_vector_store_instance
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        pipeline._is_initialized = True
        pipeline.vector_store_manager = mock_vector_store_instance
        
        await pipeline.cleanup()
        
        mock_vector_store_instance.close.assert_called_once()
        assert not pipeline.is_initialized


@pytest.mark.asyncio
async def test_pipeline_cleanup_with_provider_cleanup(mock_env_vars):
    """Tests cleanup when providers have cleanup methods."""
    pipeline = RAGPipeline(llm="openai", embed_model="openai")
    
    # Mock providers with cleanup methods
    mock_llm = MagicMock()
    mock_llm.cleanup = AsyncMock()
    pipeline.llm_provider = mock_llm
    
    mock_embedding = MagicMock()
    mock_embedding.cleanup = AsyncMock()
    pipeline.embedding_provider = mock_embedding
    
    mock_vector_store = MagicMock()
    mock_vector_store.close = AsyncMock()
    pipeline.vector_store_manager = mock_vector_store
    
    pipeline._is_initialized = True
    
    await pipeline.cleanup()
    
    mock_llm.cleanup.assert_called_once()
    mock_embedding.cleanup.assert_called_once()
    mock_vector_store.close.assert_called_once()
    assert not pipeline.is_initialized


@pytest.mark.asyncio
async def test_pipeline_cleanup_handles_errors(mock_env_vars):
    """Tests that cleanup continues even if some operations fail."""
    pipeline = RAGPipeline(llm="openai", embed_model="openai")
    
    mock_vector_store = MagicMock()
    mock_vector_store.close = AsyncMock(side_effect=Exception("Close failed"))
    pipeline.vector_store_manager = mock_vector_store
    
    pipeline._is_initialized = True
    
    # Should not raise exception
    await pipeline.cleanup()
    
    assert not pipeline.is_initialized


# Test logging configuration
def test_logging_configuration_default():
    """Tests that logging is configured with INFO level by default."""
    with patch.dict(os.environ, {}, clear=True):
        # Re-import to trigger logging configuration
        import importlib
        import yasrl.pipeline
        importlib.reload(yasrl.pipeline)
        
        logger = logging.getLogger("yasrl.pipeline")
        assert logger.level == logging.INFO or logger.level == logging.NOTSET


def test_logging_configuration_from_env(monkeypatch):
    """Tests that logging level is set from LOG_LEVEL environment variable."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    
    # Re-import to trigger logging configuration
    import importlib
    import yasrl.pipeline
    importlib.reload(yasrl.pipeline)
    
    # The root logger should have DEBUG level
    assert logging.getLogger().level == logging.DEBUG


# Test initialization statistics
@pytest.mark.asyncio
async def test_pipeline_initialization_statistics(mock_env_vars):
    """Tests that initialization timing statistics are collected."""
    with patch("yasrl.pipeline.ConfigurationManager") as mock_config_manager, \
         patch("yasrl.pipeline.LLMProviderFactory"), \
         patch("yasrl.pipeline.EmbeddingProviderFactory"), \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager:
        
        # Setup minimal mocks
        mock_config_instance = MagicMock()
        mock_config_instance.get_database_config.return_value = {"postgres_uri": "postgresql://test:test@localhost/test"}
        mock_config_instance.load_config.return_value = MagicMock(
            embedding=MagicMock(chunk_size=1024),
            chunk_overlap=200,
            retrieval_top_k=10,
            rerank_top_k=5
        )
        mock_config_instance.get_config_sources.return_value = []
        mock_config_manager.return_value = mock_config_instance
        
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.initialize = AsyncMock()
        mock_vector_store_manager.return_value = mock_vector_store_instance
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        await pipeline.initialize()
        
        stats = pipeline.get_initialization_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        assert "Configuration validated" in stats
        assert "LLM provider 'openai' created" in stats
        assert "Embedding provider 'openai' created" in stats
        assert "Database connection verified" in stats
        assert all(isinstance(v, float) for v in stats.values())


# Test QueryProcessor initialization
@pytest.mark.asyncio
async def test_query_processor_initialization(mock_env_vars):
    """Tests that QueryProcessor is properly initialized."""
    with patch("yasrl.pipeline.ConfigurationManager") as mock_config_manager, \
         patch("yasrl.pipeline.LLMProviderFactory") as mock_llm_factory, \
         patch("yasrl.pipeline.EmbeddingProviderFactory"), \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager:
        
        mock_config_instance = MagicMock()
        mock_config_instance.get_database_config.return_value = {"postgres_uri": "postgresql://test:test@localhost/test"}
        mock_config_instance.load_config.return_value = MagicMock(
            embedding=MagicMock(chunk_size=1024),
            chunk_overlap=200,
            retrieval_top_k=15,
            rerank_top_k=7
        )
        mock_config_instance.get_config_sources.return_value = []
        mock_config_manager.return_value = mock_config_instance
        
        mock_llm = MagicMock()
        mock_llm_factory.create.return_value = mock_llm
        
        mock_vector_store = MagicMock()
        mock_vector_store.initialize = AsyncMock()
        mock_vector_store_manager.return_value = mock_vector_store
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        await pipeline.initialize()
        
        assert pipeline.query_processor is not None
        assert pipeline.query_processor.llm_provider == mock_llm
        assert pipeline.query_processor.vector_store_manager == mock_vector_store
        assert pipeline.query_processor.config["retrieval_top_k"] == 15
        assert pipeline.query_processor.config["rerank_top_k"] == 7


# Test initialization failure and cleanup
@pytest.mark.asyncio
async def test_pipeline_initialization_failure_triggers_cleanup(mock_env_vars):
    """Tests that cleanup is called when initialization fails."""
    with patch("yasrl.pipeline.ConfigurationManager"), \
         patch("yasrl.pipeline.LLMProviderFactory") as mock_llm_factory, \
         patch("yasrl.pipeline.EmbeddingProviderFactory"), \
         patch("yasrl.pipeline.VectorStoreManager"):
        
        # Make LLM factory fail
        mock_llm_factory.create.side_effect = Exception("LLM creation failed")
        
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        
        with patch.object(pipeline, 'cleanup') as mock_cleanup:
            mock_cleanup.return_value = None
            
            with pytest.raises(Exception) as exc_info:
                await pipeline.initialize()
            
            assert "LLM creation failed" in str(exc_info.value)
            mock_cleanup.assert_called_once()


# Test various provider combinations
@pytest.mark.asyncio
@pytest.mark.parametrize("llm,embed", [
    ("openai", "openai"),
    ("azure", "azure"),
    ("anthropic", "huggingface"),
    ("local", "local"),
])
async def test_pipeline_various_provider_combinations(llm, embed, monkeypatch):
    """Tests initialization with various provider combinations."""
    # Set required environment variables for each provider
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_DB", "test")
    
    if llm == "openai" or embed == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    if llm == "azure" or embed == "azure":
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.azure.com")
    if llm == "anthropic":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    
    with patch("yasrl.pipeline.ConfigurationManager") as mock_config_manager, \
         patch("yasrl.pipeline.LLMProviderFactory"), \
         patch("yasrl.pipeline.EmbeddingProviderFactory"), \
         patch("yasrl.pipeline.VectorStoreManager") as mock_vector_store_manager:
        
        mock_config_instance = MagicMock()
        mock_config_instance.validate_config = MagicMock()
        mock_config_instance.get_database_config.return_value = {"postgres_uri": "postgresql://test:test@localhost/test"}
        mock_config_instance.load_config.return_value = MagicMock(
            embedding=MagicMock(chunk_size=1024),
            chunk_overlap=200,
            retrieval_top_k=10,
            rerank_top_k=5
        )
        mock_config_instance.get_config_sources.return_value = []
        mock_config_manager.return_value = mock_config_instance
        
        mock_vector_store = MagicMock()
        mock_vector_store.initialize = AsyncMock()
        mock_vector_store_manager.return_value = mock_vector_store
        
        pipeline = RAGPipeline(llm=llm, embed_model=embed)
        await pipeline.initialize()
        
        assert pipeline.is_initialized
        mock_config_instance.validate_config.assert_called_once_with(llm, embed)