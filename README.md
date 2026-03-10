# papra-llm-manager

Enhance [Papra](https://papra.app) with AI-powered features. This tool provides intelligent text extraction from images and documents using LLMs (Claude, GPT-4 Vision), and automatic tagging based on document understanding.

## Features

- **LLM-powered text extraction**: Extract text from images and documents when OCR fails
- **Intelligent tagging**: Generate and apply tags based on document content understanding
- **Batch processing**: Process multiple documents concurrently
- **Multi-provider support**: Works with Anthropic Claude and OpenAI GPT-4
- **CLI tool**: Easy-to-use command-line interface
- **Python library**: Use as a library in your own applications

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/ecwu/papra-llm-manager.git
cd papra-llm-manager

# Install with uv
uv install
```

### Using pip

```bash
pip install papra-llm-manager
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# Papra API Configuration
PAPAPRA_API_TOKEN=your_papra_api_token
PAPRA_ORG_ID=your_papra_organization_id

# LLM Provider Configuration (choose one)
LLM_PROVIDER=anthropic  # or 'openai'
LLM_API_KEY=your_llm_api_key

# Optional: Specify model
# LLM_MODEL=claude-3-5-sonnet-20241022
```

Run `init` command to generate a template `.env` file:

```bash
papra-llm init
```

## CLI Usage

### Upload a document with AI enhancements

```bash
# Upload with text extraction and auto-tagging
papra-llm upload document.pdf --extract-text --auto-tag

# Upload with OCR languages
papra-llm upload receipt.jpg --extract-text --auto-tag --ocr-languages "eng,fra"
```

### Process documents

```bash
# Process all documents with missing text
papra-llm process-missing --batch-size 10

# Process a specific document
papra-llm process --document-id doc_123

# Process all documents (extract text + auto-tag)
papra-llm process
```

### Tag documents

```bash
# Re-tag all documents using LLM
papra-llm re-tag-all --max-tags 5 --batch-size 10
```

### List and search

```bash
# List all documents
papra-llm list

# Search documents
papra-llm search "invoice"
```

## Library Usage

### Basic example

```python
import asyncio
from papra_llm_manager import Config, PapraClient, AnthropicProvider, DocumentProcessor

async def main():
    # Load configuration
    config = Config.from_env()

    # Initialize client and LLM
    client = PapraClient(api_token=config.papra_api_token)
    llm = AnthropicProvider(api_key=config.llm_api_key)

    # Create processor
    processor = DocumentProcessor(
        papra_client=client,
        llm_handler=llm,
        extract_text_threshold=100,
        max_tags=5,
    )

    # Upload and process a document
    doc = await client.upload_document(config.papra_org_id, "invoice.pdf")
    result = await processor.process_document(
        org_id=config.papra_org_id,
        document_id=doc.id,
        extract_text=True,
        generate_tags=True,
    )

    print(f"Success: {result.success}")
    print(f"Tags added: {[t.name for t in result.tags_added]}")

asyncio.run(main())
```

### Advanced usage

```python
import asyncio
from papra_llm_manager import Config, PapraClient, DocumentTagger, AnthropicProvider

async def main():
    config = Config.from_env()
    client = PapraClient(api_token=config.papra_api_token)
    llm = AnthropicProvider(api_key=config.llm_api_key)

    # Create custom tagger with custom colors
    tag_colors = {
        "invoice": "#EF4444",
        "receipt": "#F59E0B",
        "contract": "#3B82F6",
    }

    tagger = DocumentTagger(
        papra_client=client,
        llm_handler=llm,
        tag_colors=tag_colors,
    )

    # Re-tag all documents
    stats = await tagger.re_tag_all_documents(
        org_id=config.papra_org_id,
        max_tags=5,
        batch_size=10,
    )

    print(f"Processed: {stats['processed']} documents")
    print(f"Tags added: {stats['tags_added']}")

asyncio.run(main())
```

## LLM Providers

### Anthropic Claude

```python
from papra_llm_manager import AnthropicProvider

llm = AnthropicProvider(
    api_key="your_anthropic_api_key",
    model="claude-3-5-sonnet-20241022",  # Default
)
```

### OpenAI GPT-4

```python
from papra_llm_manager import OpenAIProvider

llm = OpenAIProvider(
    api_key="your_openai_api_key",
    model="gpt-4o",  # Default
)
```

### Factory function

```python
from papra_llm_manager import create_llm_provider

llm = create_llm_provider(
    provider="anthropic",  # or "openai"
    api_key="your_api_key",
    model="claude-3-5-sonnet-20241022",
)
```

## API Reference

### PapraClient

Main client for interacting with Papra API.

- `upload_document(org_id, file_path, ocr_languages)` - Upload a document
- `list_documents(org_id, page_index, page_size, tag_ids)` - List documents
- `get_document(org_id, document_id)` - Get a document
- `update_document_content(org_id, document_id, content, name)` - Update document
- `delete_document(org_id, document_id)` - Delete a document
- `search_documents(org_id, search_query, page_index, page_size)` - Search documents
- `list_tags(org_id)` - List all tags
- `create_tag(org_id, name, color, description)` - Create a tag
- `add_tag_to_document(org_id, document_id, tag_id)` - Add tag to document

### DocumentProcessor

Process documents with LLM capabilities.

- `process_document(org_id, document_id, extract_text, generate_tags)` - Process a document
- `process_missing_text(org_id, batch_size)` - Process documents with missing text
- `process_all(org_id, batch_size, extract_text, generate_tags)` - Process all documents

### DocumentTagger

Intelligent tagging system.

- `tag_document(org_id, document_id, text, document_name, max_tags)` - Tag a document
- `re_tag_all_documents(org_id, max_tags, batch_size)` - Re-tag all documents
- `sync_tags_to_papra(org_id, tag_names, tag_color)` - Sync tags to Papra

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
