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
uv install
```

### Using pip

```bash
pip install papra-llm-manager
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# Papra API
PAPRA_API_TOKEN=your_papra_api_token
PAPRA_ORG_ID=your_papra_organization_id

# LLM Configuration (provider/model format)
LLM_MODEL=anthropic/claude-3-5-sonnet-20241022
LLM_API_KEY=your_llm_api_key

# Optional: Custom API base (for Ollama, Azure, proxies)
# LLM_API_BASE=http://localhost:11434
```

Run `init` command to generate a template `.env` file:

```bash
papra-llm init
```

## Quick Start

### CLI Usage

```bash
# Upload with text extraction and auto-tagging
papra-llm upload document.pdf --extract-text --auto-tag

# Process documents with missing text
papra-llm process-missing --batch-size 10

# Re-tag all documents
papra-llm re-tag-all --max-tags 5
```

### Using Ollama (local)

```bash
# Pull a vision model
ollama pull llava
ollama pull qwen2.5vl

# Configure .env
LLM_MODEL=ollama/llava
LLM_API_BASE=http://localhost:11434
LLM_API_KEY=any-value  # Required but unused
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
| Anthropic | `anthropic/claude-3-5-sonnet-20241022` | Default |
| OpenAI | `openai/gpt-4o` | |
| Ollama | `ollama/llava` | Local, vision support |
| Azure OpenAI | `azure/gpt-4o` | Enterprise |
| Google Gemini | `gemini/gemini-pro-vision` | |
| DeepSeek | `deepseek/deepseek-chat` | |

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for full list.

## Architecture

- `LiteLLMProvider` - Unified LLM interface via [LiteLLM](https://litellm.ai)
- `DocumentProcessor` - Orchestrates text extraction and tagging
- `DocumentTagger` - AI-powered tagging system
- `PapraClient` - Async client for Papra API
- Prompts are stored in `papra_llm_manager/prompts/`

## License

MIT