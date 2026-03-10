"""Document processing logic for Papra.

This module provides functionality to process documents, including:
- Extracting text from images using LLM when OCR fails
- Generating and applying tags
- Extracting metadata
- Batch processing
"""

import asyncio
import io
from typing import TYPE_CHECKING, Callable, List, Optional

from PIL import Image

from papra_llm_manager.client import PapraClient
from papra_llm_manager.llm_handler import LLMError, LLMProvider
from papra_llm_manager.models import Document, ProcessingResult, Tag

if TYPE_CHECKING:
    from papra_llm_manager.tagger import DocumentTagger


class DocumentProcessor:
    """Process documents to extract missing text and metadata.

    This class combines Papra API client with LLM capabilities to
    automatically process documents, extract text, and apply intelligent tagging.
    """

    def __init__(
        self,
        papra_client: PapraClient,
        llm_handler: LLMProvider,
        extract_text_threshold: int = 100,
        max_tags: int = 5,
        tagger: Optional["DocumentTagger"] = None,
    ):
        """Initialize the document processor.

        Args:
            papra_client: Papra API client instance
            llm_handler: LLM provider instance
            extract_text_threshold: Text length below which to re-extract
            max_tags: Maximum number of tags to generate per document
            tagger: Optional tagger instance (created if not provided)
        """
        self.papra = papra_client
        self.llm = llm_handler
        self.extract_text_threshold = extract_text_threshold
        self.max_tags = max_tags
        
        if tagger is not None:
            self.tagger = tagger
        else:
            from papra_llm_manager.tagger import DocumentTagger
            self.tagger = DocumentTagger(
                papra_client=papra_client,
                llm_handler=llm_handler,
            )

    def _should_extract_text(self, document: Document) -> bool:
        """Determine if a document needs text extraction.

        Args:
            document: The document to check

        Returns:
            bool: True if text extraction is needed
        """
        if not document.has_text:
            return True
        return document.text_length < self.extract_text_threshold

    async def process_document(
        self,
        org_id: str,
        document_id: str,
        extract_text: bool = True,
        generate_tags: bool = True,
        extract_metadata: bool = False,
    ) -> ProcessingResult:
        """Process a single document.

        This method:
        1. Fetches the document
        2. Checks if text extraction is needed and performs it if so
        3. Generates and applies tags
        4. Extracts metadata if requested

        Args:
            org_id: Organization ID
            document_id: Document ID
            extract_text: Whether to extract missing text
            generate_tags: Whether to generate and apply tags
            extract_metadata: Whether to extract metadata

        Returns:
            ProcessingResult: Result of the processing operation
        """
        result = ProcessingResult(
            document=None,
            success=False,
        )

        try:
            # Fetch the document
            doc = await self.papra.get_document(org_id, document_id)
            result.document = doc

            # Check if text extraction is needed
            needs_text = extract_text and self._should_extract_text(doc)

            if needs_text:
                print(f"  Extracting text for document: {doc.name}")
                extracted_text = await self._extract_text_from_document(
                    org_id, document_id, doc
                )
                result.text_extracted = bool(extracted_text)

                # Update the document with extracted text
                if extracted_text:
                    await self.papra.update_document_content(
                        org_id, document_id, content=extracted_text
                    )
                    # Refresh document to get updated content
                    doc = await self.papra.get_document(org_id, document_id)
                    result.document = doc

            # Generate and apply tags
            if generate_tags and doc.has_text:
                print(f"  Generating tags for document: {doc.name}")
                tags = await self.tagger.tag_document(
                    org_id=org_id,
                    document_id=document_id,
                    text=doc.content,
                    document_name=doc.name,
                    max_tags=self.max_tags,
                )
                result.tags_added = tags

            # Extract metadata
            if extract_metadata and doc.has_text:
                try:
                    print(f"  Extracting metadata for document: {doc.name}")
                    metadata = await self._extract_metadata_from_document(doc)
                    result.metadata_extracted = metadata
                except LLMError as e:
                    print(f"  Warning: Failed to extract metadata: {e}")

            result.success = True
            return result

        except Exception as e:
            result.error = str(e)
            print(f"  Error processing document {document_id}: {e}")
            return result

    async def _extract_text_from_document(
        self, org_id: str, document_id: str, doc: Document
    ) -> Optional[str]:
        """Extract text from a document using LLM.

        Args:
            org_id: Organization ID
            document_id: Document ID
            doc: Document object

        Returns:
            Optional[str]: Extracted text or None if failed
        """
        try:
            # Download the document file
            file_content = await self.papra.get_document_file(org_id, document_id)

            # Try to open as an image
            try:
                image = Image.open(io.BytesIO(file_content))

                # If PDF with multiple pages, handle all pages
                if image.format == "PDF":
                    return await self._extract_text_from_pdf(file_content, doc)

                # Extract text using LLM
                extracted_text = await self.llm.extract_text_from_image(
                    image, document_name=doc.name
                )
                return extracted_text.strip() if extracted_text else None

            except Exception as e:
                print(f"    Warning: Could not open as image: {e}")
                return None

        except LLMError as e:
            print(f"    Warning: LLM text extraction failed: {e}")
            return None
        except Exception as e:
            print(f"    Warning: Text extraction failed: {e}")
            return None

    async def _extract_text_from_pdf(self, file_content: bytes, doc: Document) -> Optional[str]:
        """Extract text from a PDF using LLM.

        Args:
            file_content: PDF file content as bytes
            doc: Document object

        Returns:
            Optional[str]: Extracted text or None if failed
        """
        try:
            from pdf2image import convert_from_bytes

            print(f"    Converting PDF to images...")
            images = convert_from_bytes(file_content, dpi=200)

            if not images:
                print(f"    Warning: No images extracted from PDF")
                return None

            print(f"    Extracting text from {len(images)} page(s)...")
            all_text = []

            for idx, image in enumerate(images, 1):
                page_text = await self.llm.extract_text_from_image(
                    image, document_name=f"{doc.name} (page {idx})"
                )
                if page_text:
                    all_text.append(page_text.strip())

            return "\n\n".join(all_text) if all_text else None

        except ImportError:
            print(f"    Warning: pdf2image not installed. Install with: pip install pdf2image")
            return None
        except Exception as e:
            print(f"    Warning: Failed to extract text from PDF: {e}")
            return None

    async def _extract_metadata_from_document(self, doc: Document) -> dict:
        """Extract metadata from a document using LLM.

        Args:
            doc: Document object

        Returns:
            dict: Extracted metadata
        """
        return await self.llm.extract_metadata(
            text=doc.content, document_name=doc.name
        )

    async def process_missing_text(
        self,
        org_id: str,
        batch_size: int = 10,
    ) -> List[ProcessingResult]:
        """Find and process all documents with missing text content.

        Args:
            org_id: Organization ID
            batch_size: Number of documents to process concurrently

        Returns:
            List[ProcessingResult]: Results of all processing operations
        """
        print(f"Scanning for documents needing text extraction...")
        return await self._process_documents(
            org_id=org_id,
            batch_size=batch_size,
            extract_text=True,
            generate_tags=False,
            filter_func=self._should_extract_text,
        )

    async def _process_documents(
        self,
        org_id: str,
        batch_size: int,
        extract_text: bool,
        generate_tags: bool,
        filter_func: Optional[Callable[[Document], bool]] = None,
    ) -> List[ProcessingResult]:
        """Process documents with optional filtering.

        Args:
            org_id: Organization ID
            batch_size: Number of documents to process concurrently
            extract_text: Whether to extract missing text
            generate_tags: Whether to generate and apply tags
            filter_func: Optional function to filter documents

        Returns:
            List[ProcessingResult]: Results of all processing operations
        """
        all_results = []
        page_index = 0
        page_size = 100

        while True:
            documents, total_count = await self.papra.list_documents(
                org_id, page_index=page_index, page_size=page_size
            )

            if page_index == 0:
                print(f"Found {total_count} total documents")

            if not documents:
                break

            # Filter documents if filter_func provided
            to_process = (
                [doc for doc in documents if filter_func(doc)]
                if filter_func
                else documents
            )

            if to_process:
                if filter_func:
                    print(f"  Page {page_index}: {len(to_process)} documents need processing")
                else:
                    print(f"  Processing page {page_index}: {len(to_process)} documents")

                # Process in batches
                for i in range(0, len(to_process), batch_size):
                    batch = to_process[i : i + batch_size]
                    tasks = [
                        self.process_document(
                            org_id,
                            doc.id,
                            extract_text=extract_text,
                            generate_tags=generate_tags,
                        )
                        for doc in batch
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=False)
                    all_results.extend(results)

            page_index += 1

        return all_results

    async def process_all(
        self,
        org_id: str,
        batch_size: int = 10,
        extract_text: bool = True,
        generate_tags: bool = True,
    ) -> List[ProcessingResult]:
        """Process all documents in the organization.

        Args:
            org_id: Organization ID
            batch_size: Number of documents to process concurrently
            extract_text: Whether to extract missing text
            generate_tags: Whether to generate and apply tags

        Returns:
            List[ProcessingResult]: Results of all processing operations
        """
        print(f"Processing all documents in organization {org_id}...")
        return await self._process_documents(
            org_id=org_id,
            batch_size=batch_size,
            extract_text=extract_text,
            generate_tags=generate_tags,
            filter_func=None,
        )

    def print_processing_summary(self, results: List[ProcessingResult]) -> None:
        """Print a summary of processing results.

        Args:
            results: List of processing results
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        text_extracted = sum(1 for r in results if r.text_extracted)
        total_tags_added = sum(len(r.tags_added) for r in results)

        print("\n=== Processing Summary ===")
        print(f"Total documents processed: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Text extracted: {text_extracted}")
        print(f"Total tags added: {total_tags_added}")
        print(f"========================")
