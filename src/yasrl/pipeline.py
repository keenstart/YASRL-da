"""
RAG Pipeline implementation for YASRL library.
Provides the main orchestration layer for document processing and retrieval.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Optional, Any, Dict

from yasrl.config.manager import ConfigurationManager
from yasrl.exceptions import ConfigurationError
from yasrl.providers.embeddings import EmbeddingProviderFactory
from yasrl.providers.llm import LLMProviderFactory
from yasrl.text_processor import TextProcessor
from yasrl.vector_store import VectorStoreManager

# Configure logging based on environment variable
def _configure_logging():
    """Configure logging with level from environment variable."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = _configure_logging()


class QueryProcessor:
    """
    Placeholder for QueryProcessor class.
    Will be fully implemented in Phase 2 with re-ranking capabilities.
    """
    def __init__(self, llm_provider, vector_store_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize QueryProcessor.
        
        Args:
            llm_provider: The LLM provider instance
            vector_store_manager: The vector store manager instance
            config: Optional configuration dictionary
        """
        self.llm_provider = llm_provider
        self.vector_store_manager = vector_store_manager
        self.config = config or {}
        logger.debug("QueryProcessor initialized (placeholder for Phase 2)")


class RAGPipeline:
    """
    Main orchestration class for the RAG pipeline.
    
    Manages initialization and coordination of all pipeline components including:
    - Configuration management and validation
    - LLM and embedding provider setup
    - Vector store management with PostgreSQL
    - Text processing and chunking
    - Query processing (Phase 2 will add re-ranking)
    
    Supports async context manager protocol for proper resource management.
    
    Example:
        >>> # Basic initialization
        >>> pipeline = RAGPipeline(llm="openai", embed_model="openai")
        >>> await pipeline.initialize()
        >>> # ... use pipeline ...
        >>> await pipeline.cleanup()
        
        >>> # Using context manager
        >>> async with RAGPipeline(llm="openai", embed_model="openai") as pipeline:
        ...     # Pipeline is automatically initialized and cleaned up
        ...     pass
    """
    
    def __init__(self, llm: str, embed_model: str, config_file: Optional[str] = None):
        """
        Initialize the RAGPipeline.
        
        Args:
            llm: Name of the language model provider (e.g., "openai", "azure", "anthropic")
            embed_model: Name of the embedding model provider (e.g., "openai", "azure", "huggingface")
            config_file: Optional path to configuration file
            
        Raises:
            ValueError: If provider names are invalid
        """
        logger.info(f"Creating RAGPipeline with LLM: {llm}, Embedding: {embed_model}")
        
        # Validate provider names
        valid_llm_providers = {"openai", "azure", "anthropic", "local"}
        valid_embedding_providers = {"openai", "azure", "huggingface", "local"}
        
        if llm not in valid_llm_providers:
            raise ValueError(f"Invalid LLM provider: {llm}. Must be one of {valid_llm_providers}")
        if embed_model not in valid_embedding_providers:
            raise ValueError(f"Invalid embedding provider: {embed_model}. Must be one of {valid_embedding_providers}")
        
        # Store configuration
        self.llm_name = llm
        self.embed_model_name = embed_model
        self.config_file = config_file
        
        # Initialize component placeholders
        self._is_initialized = False
        self.config_manager: Optional[ConfigurationManager] = None
        self.llm_provider = None
        self.embedding_provider = None
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.text_processor: Optional[TextProcessor] = None
        self.query_processor: Optional[QueryProcessor] = None
        
        # Track initialization timing
        self._init_times = {}
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        
    def _log_init_step(self, step: str, duration: float):
        """Log initialization step timing."""
        self._init_times[step] = duration
        logger.info(f"  ✓ {step} ({duration:.3f}s)")
        
    async def initialize(self):
        """
        Asynchronously initialize all pipeline components.
        
        Performs the following steps:
        1. Validates configuration and environment variables
        2. Creates LLM and embedding provider instances
        3. Initializes vector store with database connectivity check
        4. Sets up text processor with configured chunk size
        5. Initializes query processor for retrieval
        
        Raises:
            ConfigurationError: If required configuration is missing or invalid
            ConnectionError: If database connection fails
            Exception: For other initialization failures
        """
        if self._is_initialized:
            logger.info("Pipeline already initialized, skipping.")
            return
            
        logger.info("=" * 60)
        logger.info("Initializing RAG Pipeline")
        logger.info("=" * 60)
        
        total_start = perf_counter()
        
        try:
            # Step 1: Initialize and validate configuration
            step_start = perf_counter()
            self.config_manager = ConfigurationManager(config_file=self.config_file)
            self.config_manager.validate_config(self.llm_name, self.embed_model_name)
            config = self.config_manager.load_config()
            self._log_init_step("Configuration validated", perf_counter() - step_start)
            
            # Step 2: Create LLM provider
            step_start = perf_counter()
            self.llm_provider = LLMProviderFactory.create(self.llm_name)
            self._log_init_step(f"LLM provider '{self.llm_name}' created", perf_counter() - step_start)
            
            # Step 3: Create embedding provider
            step_start = perf_counter()
            self.embedding_provider = EmbeddingProviderFactory.create(self.embed_model_name)
            self._log_init_step(f"Embedding provider '{self.embed_model_name}' created", perf_counter() - step_start)
            
            # Step 4: Initialize VectorStoreManager with database connection
            step_start = perf_counter()
            db_config = self.config_manager.get_database_config()
            
            # Use postgres_uri if available, otherwise construct from components
            if "postgres_uri" in db_config:
                self.vector_store_manager = VectorStoreManager(
                    postgres_uri=db_config["postgres_uri"],
                    vector_dimensions=db_config.get("vector_dimensions", 1536),
                    table_prefix=db_config.get("table_prefix", "yasrl")
                )
            else:
                # Construct postgres_uri from components
                postgres_uri = (
                    f"postgresql://{db_config['user']}:{db_config['password']}"
                    f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
                )
                self.vector_store_manager = VectorStoreManager(
                    postgres_uri=postgres_uri,
                    vector_dimensions=db_config.get("vector_dimensions", 1536),
                    table_prefix=db_config.get("table_prefix", "yasrl")
                )
            
            # Test database connectivity
            try:
                # VectorStoreManager might need async initialization
                if hasattr(self.vector_store_manager, 'initialize'):
                    await self.vector_store_manager.initialize()
                else:
                    # Synchronous test connection
                    self.vector_store_manager.setup_schema()
                self._log_init_step("Database connection verified", perf_counter() - step_start)
            except Exception as e:
                raise ConfigurationError(f"Failed to connect to database: {e}. "
                                       "Please check your database configuration and ensure PostgreSQL is running.")
            
            # Step 5: Set up TextProcessor
            step_start = perf_counter()
            chunk_size = config.embedding.chunk_size
            chunk_overlap = config.chunk_overlap
            self.text_processor = TextProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            self._log_init_step(f"TextProcessor initialized (chunk_size={chunk_size})", perf_counter() - step_start)
            
            # Step 6: Initialize QueryProcessor (placeholder for Phase 2)
            step_start = perf_counter()
            self.query_processor = QueryProcessor(
                self.llm_provider,
                self.vector_store_manager,
                config={
                    "retrieval_top_k": config.retrieval_top_k,
                    "rerank_top_k": config.rerank_top_k
                }
            )
            self._log_init_step("QueryProcessor initialized", perf_counter() - step_start)
            
            # Mark as initialized
            self._is_initialized = True
            
            # Log summary
            total_time = perf_counter() - total_start
            logger.info("=" * 60)
            logger.info(f"✅ Pipeline initialization complete in {total_time:.2f} seconds")
            logger.info(f"   Configuration sources: {', '.join(self.config_manager.get_config_sources())}")
            logger.info("=" * 60)
            
        except ConfigurationError as e:
            logger.error(f"❌ Configuration error during initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            # Clean up any partially initialized resources
            await self.cleanup()
            raise
            
    async def cleanup(self):
        """
        Clean up all resources and connections.
        
        Ensures proper shutdown of:
        - Database connections
        - Vector store resources
        - Any pending async operations
        """
        logger.info("Cleaning up pipeline resources...")
        
        try:
            # Close vector store connections
            if self.vector_store_manager:
                if hasattr(self.vector_store_manager, 'close'):
                    await self.vector_store_manager.close()
                logger.debug("Vector store connections closed")
                
            # Clean up providers if they have cleanup methods
            if self.llm_provider and hasattr(self.llm_provider, 'cleanup'):
                await self.llm_provider.cleanup()
                logger.debug("LLM provider cleaned up")
                
            if self.embedding_provider and hasattr(self.embedding_provider, 'cleanup'):
                await self.embedding_provider.cleanup()
                logger.debug("Embedding provider cleaned up")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Continue cleanup even if some operations fail
            
        finally:
            self._is_initialized = False
            logger.info("Pipeline cleanup complete")
            
    @asynccontextmanager
    async def managed_pipeline(self):
        """
        Context manager for automatic initialization and cleanup.
        
        Example:
            >>> async with pipeline.managed_pipeline() as p:
            ...     # Use the pipeline
            ...     pass
        """
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()
            
    @property
    def is_initialized(self) -> bool:
        """Check if the pipeline is initialized."""
        return self._is_initialized
        
    def get_initialization_stats(self) -> Dict[str, float]:
        """
        Get timing statistics from initialization.
        
        Returns:
            Dictionary mapping initialization steps to their duration in seconds
        """
        return self._init_times.copy()
