#!/usr/bin/env python3
"""
Example script demonstrating RAGPipeline initialization and configuration.
This script shows different ways to initialize the pipeline and handle errors.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import yasrl
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yasrl import RAGPipeline, ConfigurationError

# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_basic_initialization():
    """Test basic pipeline initialization with environment variables."""
    print("\n" + "=" * 60)
    print("Testing Basic Pipeline Initialization")
    print("=" * 60)
    
    # Set up test environment variables (in real usage, these would be set in your environment)
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    os.environ["POSTGRES_USER"] = "testuser"
    os.environ["POSTGRES_PASSWORD"] = "testpass"
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_DB"] = "testdb"
    
    try:
        # Create pipeline instance
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        
        # Initialize the pipeline
        await pipeline.initialize()
        
        if pipeline.is_initialized:
            print("✅ Pipeline initialized successfully!")
            
            # Get initialization statistics
            stats = pipeline.get_initialization_stats()
            print("\nInitialization Statistics:")
            for step, duration in stats.items():
                print(f"  - {step}: {duration:.3f}s")
            
            # Clean up resources
            await pipeline.cleanup()
            print("\n✅ Pipeline cleaned up successfully!")
        
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


async def test_context_manager():
    """Test pipeline with async context manager."""
    print("\n" + "=" * 60)
    print("Testing Pipeline with Context Manager")
    print("=" * 60)
    
    # Ensure environment is set up
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    os.environ["YASRL_POSTGRES_URI"] = "postgresql://testuser:testpass@localhost:5432/testdb"
    
    try:
        # Using context manager for automatic cleanup
        async with RAGPipeline(llm="openai", embed_model="openai") as pipeline:
            print("✅ Pipeline initialized via context manager")
            print(f"   Is initialized: {pipeline.is_initialized}")
            print(f"   Has LLM provider: {pipeline.llm_provider is not None}")
            print(f"   Has embedding provider: {pipeline.embedding_provider is not None}")
            print(f"   Has vector store: {pipeline.vector_store_manager is not None}")
            print(f"   Has text processor: {pipeline.text_processor is not None}")
            print(f"   Has query processor: {pipeline.query_processor is not None}")
        
        print("✅ Pipeline automatically cleaned up after context exit")
        
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


async def test_error_handling():
    """Test error handling for missing configuration."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Clear environment to trigger errors
    env_backup = dict(os.environ)
    os.environ.clear()
    
    try:
        print("\n1. Testing missing OpenAI API key:")
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        await pipeline.initialize()
        
    except ConfigurationError as e:
        print(f"   ✅ Caught expected error: {e}")
    
    # Test invalid provider
    try:
        print("\n2. Testing invalid provider name:")
        pipeline = RAGPipeline(llm="invalid_provider", embed_model="openai")
        
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")
    
    # Restore environment
    os.environ.update(env_backup)


async def test_different_providers():
    """Test initialization with different provider combinations."""
    print("\n" + "=" * 60)
    print("Testing Different Provider Combinations")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        ("openai", "openai", {"OPENAI_API_KEY": "test-key"}),
        ("azure", "azure", {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.azure.com"
        }),
        ("anthropic", "huggingface", {"ANTHROPIC_API_KEY": "test-key"}),
    ]
    
    for llm, embed, env_vars in test_configs:
        print(f"\n• Testing: LLM={llm}, Embedding={embed}")
        
        # Set up environment
        os.environ.update(env_vars)
        os.environ["YASRL_POSTGRES_URI"] = "postgresql://test:test@localhost/test"
        
        try:
            pipeline = RAGPipeline(llm=llm, embed_model=embed)
            print(f"  ✅ Pipeline created successfully")
            
            # Note: We're not actually initializing here since it would require
            # real database connections and API keys
            
        except Exception as e:
            print(f"  ❌ Error: {e}")


async def main():
    """Run all test scenarios."""
    print("\n" + "=" * 60)
    print("RAGPipeline Initialization Examples")
    print("=" * 60)
    
    # Note: These tests use mock environment variables and won't connect
    # to real services. In production, you would use actual credentials.
    
    await test_basic_initialization()
    await test_context_manager()
    await test_error_handling()
    await test_different_providers()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())