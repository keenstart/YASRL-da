__version__ = "0.1.0"

# Public API imports
from .pipeline import RAGPipeline, QueryProcessor
from .models import QueryResult, SourceChunk
from .exceptions import yasrlError, ConfigurationError, IndexingError, RetrievalError, EvaluationError
from .config.manager import ConfigurationManager
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager

__all__ = [
    "__version__",
    "RAGPipeline",
    "QueryProcessor",
    "QueryResult",
    "SourceChunk",
    "yasrlError",
    "ConfigurationError",
    "IndexingError",
    "RetrievalError",
    "EvaluationError",
    "ConfigurationManager",
    "TextProcessor",
    "VectorStoreManager",
]