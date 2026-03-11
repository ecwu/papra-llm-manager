# papra-llm-manager

Enhance [Papra](https://papra.app) with AI-powered features. Extract text from images using vision-enabled LLMs and automatically tag documents based on content understanding.

## Features

- **Vision-powered text extraction**: Extract text from images using Claude, GPT-4, or local models
- **Intelligent tagging**: Generate and apply tags based on document content
- **100+ LLM providers**: Use any LiteLLM-supported provider (Anthropic, OpenAI, Ollama, Azure, etc.)
- **Local models**: Run locally with Ollama for privacy and cost savings
- **Batch processing**: Process multiple documents concurrently
- **CLI tool**: Easy-to-use command-line interface
- **Python library**: Use as a library in your own applications

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/ecwu/papra-llm-manager.git
cd papra-llm-manager
uv sync
```

## Configuration

### Quick setup

```bash
# Initialize a .env file with template
uv run papra-llm init

# Edit .env with your credentials
# Required: PAPRA_API_TOKEN, PAPRA_ORG_ID, LLM_MODEL, LLM_API_KEY
```

### Required environment variables

- `PAPRA_API_TOKEN` - Your Papra API token
- `PAPRA_ORG_ID` - Your Papra organization ID
- `LLM_MODEL` - LLM model in `provider/model` format (e.g., `anthropic/claude-3-5-sonnet-20241022`)
- `LLM_API_KEY` - API key for your LLM provider

### Optional variables

- `PAPRA_BASE_URL` - Papra API base URL (default: `https://demo.papra.app/api`)
- `LLM_API_BASE` - Custom LLM API base (required for Ollama, Azure, etc.)
- `BATCH_SIZE` - Number of documents to process concurrently (default: 10)
- `MAX_TAGS` - Maximum tags per document (default: 5)
- `LOG_LEVEL` - Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

See `.env.example` for all available options.

### Logging

The application uses [loguru](https://github.com/Delgan/loguru) for structured logging. By default, log level is set to `INFO`, which hides detailed `DEBUG` messages.

To see more detailed logs (including debug information):
```bash
export LOG_LEVEL=DEBUG
uv run papra-llm upload document.pdf --auto-tag
```

To see only warnings and errors:
```bash
export LOG_LEVEL=WARNING
uv run papra-llm upload document.pdf --auto-tag
```

## Quick Start

### CLI Usage

```bash
# Upload with text extraction and auto-tagging
uv run papra-llm upload document.pdf --extract-text --auto-tag

# Or just auto-tag (combines both)
uv run papra-llm upload document.pdf --auto-tag

# Search documents
uv run papra-llm search "invoice"

# List all documents
uv run papra-llm list

# Process documents with missing text
uv run papra-llm process-missing --batch-size 10

# Re-tag all documents
uv run papra-llm re-tag-all --max-tags 5

# Process a specific document
uv run papra-llm process --document-id <doc-id> --extract-text --auto-tag
```

## Library Usage

### Basic example

```python
import asyncio
from papra_llm_manager import Config, PapraServiceFactory, DocumentProcessor

async def main():
    config = Config.from_env()
    client = PapraServiceFactory.create_client(config)
    llm = PapraServiceFactory.create_llm_handler(config)
    processor = DocumentProcessor(papra_client=client, llm_handler=llm)

    # Upload and process
    doc = await client.upload_document(config.papra_org_id, "invoice.pdf")
    result = await processor.process_document(
        org_id=config.papra_org_id,
        document_id=doc.id,
        extract_text=True,
        generate_tags=True,
    )
    print(f"Tags: {[t.name for t in result.tags_added]}")

asyncio.run(main())
```

### Using LiteLLM directly

```python
from papra_llm_manager import LiteLLMProvider

# Any LiteLLM-supported model
llm = LiteLLMProvider(
    model="anthropic/claude-3-5-sonnet-20241022",
    api_key="sk-...",
)

# Local Ollama
llm = LiteLLMProvider(
    model="ollama/llava",
    api_key="unused",
    api_base="http://localhost:11434",
)

# Azure OpenAI
llm = LiteLLMProvider(
    model="azure/gpt-4o",
    api_key="your-azure-key",
    api_base="https://your-resource.openai.azure.com",
)
```

## Supported Providers

| Provider | Model Format | Notes |
|----------|--------------|-------|
| Anthropic | `anthropic/claude-3-5-sonnet-20241022` | Recommended |
| DeepSeek | `deepseek/deepseek-chat` | Default, cost-effective |
| OpenAI | `openai/gpt-4o` | |
| Ollama | `ollama/llava` | Local, privacy-focused |
| Azure OpenAI | `azure/gpt-4o` | Enterprise |
| Google Gemini | `gemini/gemini-pro-vision` | |

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for 100+ supported providers.

## Architecture

- `LiteLLMProvider` - Unified LLM interface via [LiteLLM](https://litellm.ai)
- `DocumentProcessor` - Orchestrates text extraction and tagging
- `DocumentTagger` - AI-powered tagging system
- `PapraClient` - Async client for Papra API
- Prompts are stored in `papra_llm_manager/prompts/`

## License

MIT