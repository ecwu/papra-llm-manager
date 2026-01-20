"""Intelligent tagging system for Papra documents.

This module provides functionality to tag documents using LLM understanding,
syncing tags to Papra, and managing document-tag associations.
"""

import asyncio
from typing import List, Optional

from papra_llm_manager.client import PapraClient
from papra_llm_manager.llm_handler import LLMProvider, LLMError
from papra_llm_manager.models import Tag


class DocumentTagger:
    """Intelligent tagging system for documents.

    This class combines Papra API client with LLM capabilities to
    automatically generate and apply intelligent tags to documents.
    """

    def __init__(
        self,
        papra_client: PapraClient,
        llm_handler: LLMProvider,
        default_tag_color: str = "#3B82F6",
        tag_colors: Optional[dict] = None,
    ):
        """Initialize the document tagger.

        Args:
            papra_client: Papra API client instance
            llm_handler: LLM provider instance
            default_tag_color: Default hex color for new tags
            tag_colors: Optional dict mapping tag names to colors
        """
        self.papra = papra_client
        self.llm = llm_handler
        self.default_tag_color = default_tag_color
        self.tag_colors = tag_colors or {}

    def get_tag_color(self, tag_name: str) -> str:
        """Get color hex for a tag name.

        Args:
            tag_name: Name of the tag

        Returns:
            Hex color code
        """
        tag_lower = tag_name.lower()
        return self.tag_colors.get(tag_lower, self.default_tag_color)

    async def sync_tags_to_papra(
        self,
        org_id: str,
        tag_names: List[str],
        tag_color: Optional[str] = None,
    ) -> List[Tag]:
        """Create tags in Papra if they don't exist.

        Args:
            org_id: Organization ID
            tag_names: List of tag names to ensure exist
            tag_color: Optional override color for all tags

        Returns:
            List[Tag]: The synced tags
        """
        existing_tags = await self.papra.list_tags(org_id)
        existing_tag_map = {tag.name.lower(): tag for tag in existing_tags}

        synced_tags = []
        for tag_name in tag_names:
            tag_lower = tag_name.lower()
            if tag_lower in existing_tag_map:
                synced_tags.append(existing_tag_map[tag_lower])
            else:
                color = tag_color or self.get_tag_color(tag_name)
                new_tag = await self.papra.create_tag(org_id, tag_name, color)
                synced_tags.append(new_tag)

        return synced_tags

    async def generate_tags_for_document(
        self,
        text: str,
        document_name: str,
        max_tags: int = 5,
        existing_tags: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate tags using LLM for a document.

        Args:
            text: Document text content
            document_name: Document name for context
            max_tags: Maximum number of tags to generate
            existing_tags: List of existing tags to avoid duplicates

        Returns:
            List[str]: Generated tag names
        """
        try:
            tags = await self.llm.generate_tags(
                text=text,
                document_name=document_name,
                existing_tags=existing_tags,
                max_tags=max_tags,
            )
            return tags
        except LLMError as e:
            # Log error but don't fail - return empty list
            print(f"Warning: Failed to generate tags: {e}")
            return []

    async def tag_document(
        self,
        org_id: str,
        document_id: str,
        text: str,
        document_name: str,
        max_tags: int = 5,
        tag_color: Optional[str] = None,
    ) -> List[Tag]:
        """Tag a document using LLM understanding.

        This method:
        1. Fetches existing tags for document
        2. Generates new tags using LLM
        3. Creates missing tags in Papra
        4. Associates tags with document

        Args:
            org_id: Organization ID
            document_id: Document ID
            text: Document text content
            document_name: Document name
            max_tags: Maximum number of tags to generate
            tag_color: Optional override color for new tags

        Returns:
            List[Tag]: Tags associated with document
        """
        # Get existing tags for document
        doc = None
        existing_tag_names = []
        existing_tag_ids = {}
        try:
            doc = await self.papra.get_document(org_id, document_id)
            existing_tag_names = [tag.name.lower() for tag in doc.tags]
            existing_tag_ids = {tag.name.lower(): tag.id for tag in doc.tags}
        except Exception as e:
            print(f"Warning: Could not fetch document: {e}")

        # Get all tags in org for context
        org_tags = await self.papra.list_tags(org_id)
        all_tag_names = [tag.name for tag in org_tags]

        # Generate new tags
        generated_tag_names = await self.generate_tags_for_document(
            text=text,
            document_name=document_name,
            max_tags=max_tags,
            existing_tags=all_tag_names,
        )

        # Filter out tags document already has
        new_tag_names = [
            name for name in generated_tag_names if name.lower() not in existing_tag_names
        ]

        if not new_tag_names:
            return doc.tags if doc else []

        # Sync new tags to Papra
        new_tags = await self.sync_tags_to_papra(
            org_id=org_id, tag_names=new_tag_names, tag_color=tag_color
        )

        # Associate tags with document
        for tag in new_tags:
            await self.papra.add_tag_to_document(org_id, document_id, tag.id)

        return (doc.tags if doc else []) + new_tags

    async def _tag_document_with_obj(
        self,
        org_id: str,
        doc,
        max_tags: int = 5,
        tag_color: Optional[str] = None,
    ) -> List[Tag]:
        """Tag a document using LLM understanding with pre-fetched document.

        Args:
            org_id: Organization ID
            doc: Document object (already fetched with content)
            max_tags: Maximum number of tags to generate
            tag_color: Optional override color for new tags

        Returns:
            List[Tag]: Tags associated with document
        """
        existing_tag_names = [tag.name.lower() for tag in doc.tags]

        # Get all tags in org for context
        org_tags = await self.papra.list_tags(org_id)
        all_tag_names = [tag.name for tag in org_tags]

        # Generate new tags
        generated_tag_names = await self.generate_tags_for_document(
            text=doc.content,
            document_name=doc.name,
            max_tags=max_tags,
            existing_tags=all_tag_names,
        )

        # Filter out tags document already has
        new_tag_names = [
            name for name in generated_tag_names if name.lower() not in existing_tag_names
        ]

        if not new_tag_names:
            return doc.tags

        # Sync new tags to Papra
        new_tags = await self.sync_tags_to_papra(
            org_id=org_id, tag_names=new_tag_names, tag_color=tag_color
        )

        # Associate tags with document
        for tag in new_tags:
            await self.papra.add_tag_to_document(org_id, doc.id, tag.id)

        return doc.tags + new_tags

    async def re_tag_all_documents(
        self,
        org_id: str,
        max_tags: int = 5,
        batch_size: int = 10,
    ) -> dict:
        """Re-tag all documents in organization using LLM.

        Args:
            org_id: Organization ID
            max_tags: Maximum number of tags per document
            batch_size: Number of documents to process concurrently

        Returns:
            dict: Statistics about re-tagging operation
        """
        stats = {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "tags_added": 0,
        }

        # Fetch all documents
        page_index = 0
        page_size = 100

        while True:
            documents, total_count = await self.papra.list_documents(
                org_id, page_index=page_index, page_size=page_size
            )

            if page_index == 0:
                stats["total"] = total_count

            if not documents:
                break

            # Fetch full documents with content
            full_documents = []
            for doc in documents:
                try:
                    full_doc = await self.papra.get_document(org_id, doc.id)
                    full_documents.append(full_doc)
                except Exception as e:
                    print(f"Warning: Could not fetch document {doc.id}: {e}")

            # Process in batches
            for i in range(0, len(full_documents), batch_size):
                batch = full_documents[i : i + batch_size]
                await self._process_tag_batch(
                    org_id,
                    batch,
                    max_tags,
                    stats,
                )

            page_index += 1

        return stats

    async def _process_tag_batch(
        self,
        org_id: str,
        documents: List,
        max_tags: int,
        stats: dict,
    ) -> None:
        """Process a batch of documents for tagging.

        Args:
            org_id: Organization ID
            documents: List of documents to process
            max_tags: Maximum number of tags per document
            stats: Statistics dict to update
        """
        tasks = []
        for doc in documents:
            if not doc.has_text:
                stats["skipped"] += 1
                continue

            tasks.append(
                self._safe_tag_document(
                    org_id, doc, max_tags, stats
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_tag_document(
        self,
        org_id: str,
        doc,
        max_tags: int,
        stats: dict,
    ) -> None:
        """Safely tag a document with error handling.

        Args:
            org_id: Organization ID
            doc: Document to tag
            max_tags: Maximum number of tags
            stats: Statistics dict to update
        """
        try:
            old_tag_count = len(doc.tags)
            tags = await self._tag_document_with_obj(
                org_id=org_id,
                doc=doc,
                max_tags=max_tags,
            )
            stats["processed"] += 1
            stats["tags_added"] += len(tags) - old_tag_count
        except Exception as e:
            print(f"Error tagging document {doc.id} ({doc.name}): {e}")
            stats["errors"] += 1
