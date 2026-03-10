"""CLI interface for papra-llm-manager."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from papra_llm_manager.client import PapraClient, PapraClientError
from papra_llm_manager.config import Config
from papra_llm_manager.services import PapraServiceFactory
from papra_llm_manager.processors import DocumentProcessor
from papra_llm_manager.tagger import DocumentTagger


@click.group()
@click.version_option()
def cli():
    """Papra LLM Manager - Enhance Papra with AI-powered features."""
    pass


def get_config(org_id: Optional[str] = None) -> Config:
    """Load configuration from environment."""
    try:
        config = Config.from_env()
        if org_id:
            config.papra_org_id = org_id
        return config
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        click.echo(
            "Make sure PAPRA_API_TOKEN and PAPRA_ORG_ID are set in .env or environment",
            err=True,
        )
        sys.exit(1)


def get_papra_client(config: Config) -> PapraClient:
    """Create Papra client from config."""
    return PapraClient(
        api_token=config.papra_api_token,
        base_url=config.papra_base_url,
    )


def get_llm_handler(config: Config):
    """Create LLM handler from config."""
    return PapraServiceFactory.create_llm_handler(config)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--org-id", help="Papra organization ID (overrides PAPRA_ORG_ID)")
@click.option("--extract-text", is_flag=True, help="Use LLM to extract text if needed")
@click.option("--auto-tag", is_flag=True, help="Auto-tag the document using LLM understanding")
@click.option("--ocr-languages", help="OCR languages for initial extraction (comma-separated)")
def upload(file_path, org_id, extract_text, auto_tag, ocr_languages):
    """Upload a document with optional AI enhancements."""
    asyncio.run(async_upload(file_path, org_id, extract_text, auto_tag, ocr_languages))


async def async_upload(file_path, org_id, extract_text, auto_tag, ocr_languages):
    config = get_config(org_id)
    click.echo(f"Uploading {file_path} to organization {config.papra_org_id}...")
    client = get_papra_client(config)

    try:
        ocr_langs = [lang.strip() for lang in ocr_languages.split(",")] if ocr_languages else None
        doc = await client.upload_document(config.papra_org_id, file_path, ocr_languages=ocr_langs)
        click.echo(f"Uploaded document: {doc.name} (ID: {doc.id})")
        click.echo(f"  Size: {doc.size} bytes")
    except PapraClientError as e:
        click.echo(f"Upload failed: {e}", err=True)
        sys.exit(1)

    if extract_text or auto_tag:
        llm = get_llm_handler(config)
        processor = DocumentProcessor(
            papra_client=client,
            llm_handler=llm,
            extract_text_threshold=config.extract_text_threshold,
            max_tags=config.max_tags,
        )
        click.echo("Processing with AI...")
        result = await processor.process_document(
            org_id=config.papra_org_id,
            document_id=doc.id,
            extract_text=extract_text,
            generate_tags=auto_tag,
        )
        if result.success:
            click.echo("Processing complete!")
            if result.document:
                click.echo(f"  - Text content: {len(result.document.content)} characters")
            if result.text_extracted:
                click.echo("  - Text extracted from image using LLM")
            if result.tags_added:
                click.echo(f"  - Tags added: {', '.join(t.name for t in result.tags_added)}")
        else:
            click.echo(f"Processing failed: {result.error}", err=True)
            sys.exit(1)
    else:
        # If no AI processing, show text content from upload response
        # Note: This may show 0 if Papra is still processing the document
        click.echo(f"  Text content: {len(doc.content)} characters (may still be processing)")



@cli.command()
@click.option("--org-id", help="Papra organization ID (overrides PAPRA_ORG_ID)")
@click.option("--batch-size", default=10, help="Number of documents to process concurrently")
def process_missing(org_id, batch_size):
    """Process all documents with missing text content."""
    asyncio.run(async_process_missing(org_id, batch_size))


async def async_process_missing(org_id, batch_size):
    config = get_config(org_id)
    click.echo(f"Processing documents with missing text in organization {config.papra_org_id}...")
    client = get_papra_client(config)
    llm = get_llm_handler(config)
    processor = DocumentProcessor(
        papra_client=client,
        llm_handler=llm,
        extract_text_threshold=config.extract_text_threshold,
        max_tags=config.max_tags,
    )
    results = await processor.process_missing_text(
        org_id=config.papra_org_id,
        batch_size=batch_size,
    )
    processor.print_processing_summary(results)


@cli.command()
@click.option("--org-id", help="Papra organization ID (overrides PAPRA_ORG_ID)")
@click.option("--batch-size", default=10, help="Number of documents to process concurrently")
@click.option("--max-tags", default=5, help="Maximum number of tags per document")
def re_tag_all(org_id, batch_size, max_tags):
    """Re-tag all documents using LLM understanding."""
    asyncio.run(async_re_tag_all(org_id, batch_size, max_tags))


async def async_re_tag_all(org_id, batch_size, max_tags):
    config = get_config(org_id)
    click.echo(f"Re-tagging all documents in organization {config.papra_org_id}...")
    client = get_papra_client(config)
    llm = get_llm_handler(config)
    from papra_llm_manager.tagger import DocumentTagger
    tagger = DocumentTagger(
        papra_client=client,
        llm_handler=llm,
        tag_colors=config.tag_colors,
    )
    stats = await tagger.re_tag_all_documents(
        org_id=config.papra_org_id,
        max_tags=max_tags,
        batch_size=batch_size,
    )
    click.echo("\n=== Re-tagging Summary ===")
    click.echo(f"Total documents: {stats['total']}")
    click.echo(f"Processed: {stats['processed']}")
    click.echo(f"Skipped (no text): {stats['skipped']}")
    click.echo(f"Errors: {stats['errors']}")
    click.echo(f"Total tags added: {stats['tags_added']}")
    click.echo("===========================")


@cli.command()
@click.option("--org-id", help="Papra organization ID (overrides PAPRA_ORG_ID)")
@click.option("--document-id", help="Process a specific document ID")
@click.option("--extract-text", is_flag=True, help="Extract text if missing")
@click.option("--auto-tag", is_flag=True, help="Generate and apply tags")
def process(org_id, document_id, extract_text, auto_tag):
    """Process a document or all documents with AI enhancements."""
    asyncio.run(async_process(org_id, document_id, extract_text, auto_tag))


async def async_process(org_id, document_id, extract_text, auto_tag):
    config = get_config(org_id)
    if not extract_text and not auto_tag:
        extract_text = True
        auto_tag = True
    client = get_papra_client(config)
    llm = get_llm_handler(config)
    processor = DocumentProcessor(
        papra_client=client,
        llm_handler=llm,
        extract_text_threshold=config.extract_text_threshold,
        max_tags=config.max_tags,
    )
    if document_id:
        click.echo(f"Processing document {document_id}...")
        result = await processor.process_document(
            org_id=config.papra_org_id,
            document_id=document_id,
            extract_text=extract_text,
            generate_tags=auto_tag,
        )
        if result.success:
            click.echo("Processing complete!")
            if result.text_extracted:
                click.echo("  - Text extracted from image")
            if result.tags_added:
                click.echo(f"  - Tags added: {', '.join(t.name for t in result.tags_added)}")
        else:
            click.echo(f"Processing failed: {result.error}", err=True)
            sys.exit(1)
    else:
        results = await processor.process_all(
            org_id=config.papra_org_id,
            batch_size=config.batch_size,
            extract_text=extract_text,
            generate_tags=auto_tag,
        )
        processor.print_processing_summary(results)


@cli.command()
@click.option("--org-id", help="Papra organization ID (overrides PAPRA_ORG_ID)")
@click.option("--show-content", is_flag=True, help="Fetch full document content to show text length")
def list(org_id, show_content):
    """List all documents in the organization."""
    asyncio.run(async_list(org_id, show_content))


async def async_list(org_id, show_content):
    config = get_config(org_id)
    click.echo(f"Listing documents in organization {config.papra_org_id}...")
    client = get_papra_client(config)
    try:
        documents, total = await client.list_documents(config.papra_org_id)
        click.echo(f"Total documents: {total}")
        click.echo()
        for doc in documents[:50]:
            tags_str = ", ".join(t.name for t in doc.tags) if doc.tags else "no tags"
            click.echo(f"  {doc.name}")
            click.echo(f"    ID: {doc.id}")
            if show_content:
                full_doc = await client.get_document(config.papra_org_id, doc.id)
                click.echo(f"    Text: {len(full_doc.content)} chars | Tags: {tags_str}")
            else:
                click.echo(f"    Size: {doc.size} bytes | Tags: {tags_str}")
        if total > 50:
            click.echo(f"\n... and {total - 50} more documents")
    except PapraClientError as e:
        click.echo(f"Failed to list documents: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--org-id", help="Papra organization ID (overrides PAPRA_ORG_ID)")
@click.argument("query")
def search(org_id, query):
    """Search documents in the organization."""
    asyncio.run(async_search(org_id, query))


async def async_search(org_id, query):
    config = get_config(org_id)
    click.echo(f"Searching for '{query}' in organization {config.papra_org_id}...")
    client = get_papra_client(config)
    try:
        documents, total = await client.search_documents(config.papra_org_id, query)
        click.echo(f"Found {total} matching documents")
        click.echo()
        for doc in documents:
            tags_str = ", ".join(t.name for t in doc.tags) if doc.tags else "no tags"
            click.echo(f"  {doc.name}")
            click.echo(f"    ID: {doc.id} | Tags: {tags_str}")
    except PapraClientError as e:
        click.echo(f"Search failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def init():
    """Initialize a .env file with required configuration."""
    env_file = Path(".env")
    if env_file.exists():
        if not click.confirm(".env file already exists. Overwrite?"):
            return
    click.echo("Creating .env file...")
    env_content = """# Papra API Configuration
PAPRA_API_TOKEN=your_papra_api_token_here
PAPRA_ORG_ID=your_papra_organization_id_here

# Optional: Papra base URL (default: https://api.papra.app/api)
# PAPRA_BASE_URL=https://api.papra.app/api

# LLM Provider Configuration
# Choose: anthropic, openai, or deepseek (default: deepseek)
LLM_PROVIDER=deepseek

# DeepSeek uses DEEPSEEK_API_KEY or SILICONFLOW_API_KEY
# For DeepSeek: use DEEPSEEK_API_KEY
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# For SiliconFlow (OpenAI-compatible): use SILICONFLOW_API_KEY
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# Alternative generic key (also works):
LLM_API_KEY=your_llm_api_key_here

# Optional: LLM model name (uses default if not specified)
# For Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229
# For OpenAI: gpt-4o, gpt-4-turbo
# For DeepSeek text: deepseek-chat
# LLM_MODEL=deepseek-chat

# Processing Configuration
# BATCH_SIZE=10
# MAX_TAGS=5
# EXTRACT_TEXT_THRESHOLD=100
"""
    env_file.write_text(env_content)
    click.echo("Created .env file. Please edit it with your API tokens and organization ID.")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
