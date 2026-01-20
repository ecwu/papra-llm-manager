"""Advanced workflow examples for papra-llm-manager.

This example demonstrates more advanced features like batch processing,
custom tagging workflows, and metadata extraction.
"""

import asyncio

from papra_llm_manager import (
    Config,
    PapraClient,
    AnthropicProvider,
    OpenAIProvider,
    DocumentProcessor,
    DocumentTagger,
    create_llm_provider,
)


async def example_batch_processing():
    """Batch process all documents in an organization."""
    print("=== Batch Processing ===")

    config = Config.from_env()
    client = PapraClient(api_token=config.papra_api_token)
    llm = create_llm_provider(
        provider=config.llm_provider,
        api_key=config.llm_api_key,
    )

    processor = DocumentProcessor(
        papra_client=client,
        llm_handler=llm,
    )

    # Process all documents
    results = await processor.process_all(
        org_id=config.papra_org_id,
        batch_size=10,
        extract_text=True,
        generate_tags=True,
    )

    processor.print_processing_summary(results)


async def example_custom_tagging():
    """Custom tagging workflow with specific colors."""
    print("=== Custom Tagging ===")

    config = Config.from_env()
    client = PapraClient(api_token=config.papra_api_token)
    llm = AnthropicProvider(api_key=config.llm_api_key)

    # Custom tag colors
    tag_colors = {
        "invoice": "#EF4444",
        "receipt": "#F59E0B",
        "contract": "#3B82F6",
        "report": "#10B981",
    }

    tagger = DocumentTagger(
        papra_client=client,
        llm_handler=llm,
        tag_colors=tag_colors,
    )

    # Re-tag all documents
    stats = await tagger.re_tag_all_documents(
        org_id=config.papra_org_id,
        max_tags=3,
        batch_size=5,
    )

    print(f"Processed: {stats['processed']} documents")
    print(f"Tags added: {stats['tags_added']}")


async def example_metadata_extraction():
    """Extract metadata from documents."""
    print("=== Metadata Extraction ===")

    config = Config.from_env()
    client = PapraClient(api_token=config.papra_api_token)
    llm = AnthropicProvider(api_key=config.llm_api_key)

    # Get documents
    documents, _ = await client.list_documents(config.papra_org_id)

    for doc in documents[:3]:
        if not doc.has_text:
            continue

        print(f"\nDocument: {doc.name}")

        # Extract metadata
        metadata = await llm.extract_metadata(
            text=doc.content,
            document_name=doc.name,
        )

        print("Metadata:")
        for key, value in metadata.items():
            if value:
                print(f"  {key}: {value}")


async def example_search_and_process():
    """Search for documents and process specific ones."""
    print("=== Search and Process ===")

    config = Config.from_env()
    client = PapraClient(api_token=config.papra_api_token)
    llm = AnthropicProvider(api_key=config.llm_api_key)

    processor = DocumentProcessor(
        papra_client=client,
        llm_handler=llm,
    )

    # Search for specific documents
    documents, total = await client.search_documents(
        org_id=config.papra_org_id,
        search_query="invoice",
    )

    print(f"Found {total} documents matching 'invoice'")

    # Process only the first few
    for doc in documents[:5]:
        print(f"Processing: {doc.name}")
        result = await processor.process_document(
            org_id=config.papra_org_id,
            document_id=doc.id,
            generate_tags=True,
        )
        print(f"  Success: {result.success}")


async def example_multiple_providers():
    """Use different LLM providers for different tasks."""
    print("=== Multiple LLM Providers ===")

    config = Config.from_env()
    client = PapraClient(api_token=config.papra_api_token)

    # Use Claude for text extraction (better accuracy)
    claude = AnthropicProvider(api_key=config.llm_api_key)

    # Use GPT-4 for tagging (faster for this task)
    gpt4 = OpenAIProvider(api_key=config.llm_api_key)

    # Example: Extract text with Claude, tag with GPT-4
    documents, _ = await client.list_documents(config.papra_org_id)

    for doc in documents[:1]:
        if not doc.has_text:
            # Extract text with Claude
            print(f"Extracting text from {doc.name} with Claude...")
            # (Implementation would download file and call llm.extract_text_from_image)

        # Tag with GPT-4
        print(f"Tagging {doc.name} with GPT-4...")
        tags = await gpt4.generate_tags(
            text=doc.content,
            document_name=doc.name,
            max_tags=5,
        )
        print(f"  Generated tags: {tags}")


async def main():
    """Run all examples."""
    print("Papra LLM Manager - Advanced Examples\n")

    await example_batch_processing()
    await example_custom_tagging()
    await example_metadata_extraction()
    await example_search_and_process()
    await example_multiple_providers()


if __name__ == "__main__":
    asyncio.run(main())
