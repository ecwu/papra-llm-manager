"""Basic usage examples for papra-llm-manager.

This example demonstrates the core functionality of the library.
"""

import asyncio

from papra_llm_manager import (
    Config,
    PapraClient,
    AnthropicProvider,
    DocumentProcessor,
)


async def main():
    # Load configuration from environment variables
    config = Config.from_env()

    # Initialize the Papra client
    client = PapraClient(
        api_token=config.papra_api_token,
        base_url=config.papra_base_url,
    )

    # Initialize the LLM provider (using Anthropic Claude)
    llm = AnthropicProvider(
        api_key=config.llm_api_key,
        model="claude-3-5-sonnet-20241022",
    )

    # Create a document processor
    processor = DocumentProcessor(
        papra_client=client,
        llm_handler=llm,
        extract_text_threshold=config.extract_text_threshold,
        max_tags=config.max_tags,
    )

    # Example 1: Upload and process a document
    print("=== Example 1: Upload and process a document ===")
    doc = await client.upload_document(
        org_id=config.papra_org_id,
        file_path="example.pdf",
    )
    print(f"Uploaded: {doc.name} (ID: {doc.id})")

    # Process the document (extract text and generate tags)
    result = await processor.process_document(
        org_id=config.papra_org_id,
        document_id=doc.id,
        extract_text=True,
        generate_tags=True,
    )

    print(f"Success: {result.success}")
    print(f"Text extracted: {result.text_extracted}")
    print(f"Tags added: {[t.name for t in result.tags_added]}")

    # Example 2: Process documents with missing text
    print("\n=== Example 2: Process documents with missing text ===")
    results = await processor.process_missing(
        org_id=config.papra_org_id,
        batch_size=5,
    )

    processor.print_processing_summary(results)

    # Example 3: List documents
    print("\n=== Example 3: List documents ===")
    documents, total = await client.list_documents(config.papra_org_id)
    print(f"Total documents: {total}")
    for doc in documents[:5]:
        print(f"  - {doc.name} ({len(doc.content)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
