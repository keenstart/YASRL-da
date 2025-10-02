# YASRL Pipeline Examples

This directory contains example scripts demonstrating how to use the YASRL RAGPipeline.

## Examples

### test_initialization.py

Demonstrates various ways to initialize and configure the RAGPipeline:

- Basic initialization with environment variables
- Using async context managers for automatic resource cleanup
- Error handling for missing configuration
- Different provider combinations (OpenAI, Azure, Anthropic, HuggingFace)

## Running the Examples

### Prerequisites

1. Install the required dependencies:
```bash
pip install -r ../requirements.txt
```

2. Set up environment variables:
```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# For PostgreSQL database
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="your-database"

# Or use a single connection URI
export YASRL_POSTGRES_URI="postgresql://user:pass@localhost:5432/dbname"
```

3. Ensure PostgreSQL is running with the pgvector extension installed.

### Running Examples

```bash
# Run initialization examples
python examples/test_initialization.py
```

## Configuration Options

The RAGPipeline supports multiple configuration methods:

1. **Environment Variables** (highest priority)
   - Individual database parameters: `POSTGRES_USER`, `POSTGRES_PASSWORD`, etc.
   - Connection URI: `YASRL_POSTGRES_URI`
   - Provider API keys: `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, etc.

2. **Configuration Files**
   - Local: `yasrl.yaml` or `yasrl.json` in current directory
   - Global: `~/.yasrl/config.yaml`
   - Custom: Pass `config_file` parameter to RAGPipeline

3. **Default Values** (lowest priority)
   - Sensible defaults for chunk sizes, timeouts, etc.

## Error Handling

The pipeline provides clear error messages for common configuration issues:

- Missing API keys for chosen providers
- Incomplete database configuration
- Invalid provider names
- Database connection failures

## Context Manager Usage

The RAGPipeline supports Python's async context manager protocol:

```python
async with RAGPipeline(llm="openai", embed_model="openai") as pipeline:
    # Pipeline is automatically initialized
    # Use the pipeline here
    pass
# Pipeline is automatically cleaned up
```

## Initialization Statistics

The pipeline tracks initialization timing for performance monitoring:

```python
pipeline = RAGPipeline(llm="openai", embed_model="openai")
await pipeline.initialize()

stats = pipeline.get_initialization_stats()
# Returns dict with timing for each initialization step
```