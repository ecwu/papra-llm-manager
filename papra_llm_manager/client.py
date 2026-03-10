"""Papra API client for interacting with the Papra service."""

import asyncio
import mimetypes
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import aiofiles
import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from papra_llm_manager.models import (
    ApiKey,
    Document,
    Organization,
    OrganizationStats,
    Tag,
)


class PapraClientError(Exception):
    """Base exception for Papra client errors."""

    pass


class PapraAuthenticationError(PapraClientError):
    """Exception for authentication errors."""

    pass


class PapraNotFoundError(PapraClientError):
    """Exception for resource not found errors."""

    pass


class PapraClient:
    """Client for interacting with Papra API.

    This client provides async methods for all major Papra API operations.
    """

    def __init__(
        self,
        api_token: str,
        base_url: str = "https://api.papra.app/api",
        timeout: float = 30.0,
    ):
        """Initialize the Papra client.

        Args:
            api_token: Papra API bearer token
            base_url: Base URL for the Papra API
            timeout: Request timeout in seconds
        """
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self._headers(),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _headers(self) -> dict:
        """Get request headers with authentication."""
        return {"Authorization": f"Bearer {self.api_token}"}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
        ),
        reraise=True,
    )
    async def _execute_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute request with retries for transient HTTP errors."""
        response = await self.client.request(method, url, **kwargs)
        if response.status_code in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        return response

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with proper error handling.

        Args:
            method: HTTP method
            path: API path
            **kwargs: Additional arguments for httpx.request

        Returns:
            httpx.Response

        Raises:
            PapraAuthenticationError: If authentication fails
            PapraNotFoundError: If resource not found
            PapraClientError: For other API errors
        """
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", None)
        if headers:
            kwargs["headers"] = headers

        try:
            response = await self._execute_request(method, url, **kwargs)
        except httpx.TimeoutException:
            raise PapraClientError(f"Request timeout: {url}")
        except httpx.NetworkError as e:
            raise PapraClientError(f"Network error: {str(e)}")
        except httpx.HTTPError as e:
            raise PapraClientError(f"HTTP error: {str(e)}")

        if response.status_code == 401:
            raise PapraAuthenticationError("Invalid or expired API token")
        elif response.status_code == 404:
            raise PapraNotFoundError(f"Resource not found: {url}")
        elif response.status_code >= 400:
            error_msg = response.text or f"HTTP {response.status_code}"
            raise PapraClientError(f"API error: {error_msg}")

        return response

    # API Key operations
    async def get_current_api_key(self) -> ApiKey:
        """Get information about the currently used API key.

        Returns:
            ApiKey: The current API key information
        """
        response = await self._request("GET", "/api-keys/current")
        data = response.json()["apiKey"]
        return ApiKey(
            id=data["id"],
            name=data["name"],
            permissions=data["permissions"],
        )

    # Organization operations
    async def list_organizations(self) -> List[Organization]:
        """List all organizations accessible to the authenticated user.

        Returns:
            List[Organization]: List of organizations
        """
        response = await self._request("GET", "/organizations")
        orgs_data = response.json()["organizations"]

        return [
            Organization(
                id=org["id"],
                name=org["name"],
                created_at=datetime.fromisoformat(org["createdAt"]),
                updated_at=datetime.fromisoformat(org["updatedAt"]),
            )
            for org in orgs_data
        ]

    async def get_organization(self, organization_id: str) -> Organization:
        """Get an organization by its ID.

        Args:
            organization_id: The organization ID

        Returns:
            Organization: The organization
        """
        response = await self._request("GET", f"/organizations/{organization_id}")
        org = response.json()["organization"]

        return Organization(
            id=org["id"],
            name=org["name"],
            created_at=datetime.fromisoformat(org["createdAt"]),
            updated_at=datetime.fromisoformat(org["updatedAt"]),
        )

    async def create_organization(self, name: str) -> Organization:
        """Create a new organization.

        Args:
            name: The organization name (3-50 characters)

        Returns:
            Organization: The created organization
        """
        response = await self._request("POST", "/organizations", json={"name": name})
        org = response.json()["organization"]

        return Organization(
            id=org["id"],
            name=org["name"],
            created_at=datetime.fromisoformat(org["createdAt"]),
            updated_at=datetime.fromisoformat(org["updatedAt"]),
        )

    async def update_organization(
        self, organization_id: str, name: str
    ) -> Organization:
        """Update an organization's name.

        Args:
            organization_id: The organization ID
            name: The new organization name (3-50 characters)

        Returns:
            Organization: The updated organization
        """
        response = await self._request(
            "PUT", f"/organizations/{organization_id}", json={"name": name}
        )
        org = response.json()["organization"]

        return Organization(
            id=org["id"],
            name=org["name"],
            created_at=datetime.fromisoformat(org["createdAt"]),
            updated_at=datetime.fromisoformat(org["updatedAt"]),
        )

    async def delete_organization(self, organization_id: str) -> None:
        """Delete an organization by its ID.

        Args:
            organization_id: The organization ID
        """
        await self._request("DELETE", f"/organizations/{organization_id}")

    # Document operations
    def _parse_document(self, data: dict) -> Document:
        """Parse document data from API response."""
        tags = []
        if "tags" in data and data["tags"]:
            tags = [
                Tag(
                    id=tag["id"],
                    name=tag["name"],
                    color=tag["color"],
                    description=tag.get("description"),
                )
                for tag in data["tags"]
            ]

        return Document(
            id=data["id"],
            name=data["name"],
            organization_id=data["organizationId"],
            content=data.get("content", ""),
            size=data.get("originalSize", 0),
            mime_type=data.get("mimeType", ""),
            created_at=datetime.fromisoformat(data["createdAt"]),
            updated_at=datetime.fromisoformat(data["updatedAt"]),
            tags=tags,
            file_hash=data.get("originalSha256Hash"),
        )

    async def upload_document(
        self,
        org_id: str,
        file_path: str,
        ocr_languages: Optional[List[str]] = None,
    ) -> Document:
        """Upload a new document to the organization.

        Args:
            org_id: The organization ID
            file_path: Path to the file to upload
            ocr_languages: Optional list of OCR languages

        Returns:
            Document: The created document
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with aiofiles.open(path, "rb") as f:
            file_data = await f.read()

        # Detect MIME type from file extension
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None:
            mime_type = "application/octet-stream"

        files = {"file": (path.name, file_data, mime_type)}
        data = {}
        if ocr_languages:
            data["ocrLanguages"] = ",".join(ocr_languages)

        response = await self._request(
            "POST", f"/organizations/{org_id}/documents", files=files, data=data
        )
        doc = response.json()["document"]

        return self._parse_document(doc)

    async def list_documents(
        self,
        org_id: str,
        page_index: int = 0,
        page_size: int = 100,
        tag_ids: Optional[List[str]] = None,
    ) -> tuple[List[Document], int]:
        """List all documents in the organization.

        Args:
            org_id: The organization ID
            page_index: The page index to start from (default: 0)
            page_size: The number of documents to return (default: 100)
            tag_ids: Optional list of tag IDs to filter by

        Returns:
            Tuple[List[Document], int]: List of documents and total count
        """
        params: dict = {"pageIndex": page_index, "pageSize": page_size}
        if tag_ids:
            params["tags"] = ",".join(tag_ids)

        response = await self._request(
            "GET", f"/organizations/{org_id}/documents", params=params
        )
        result = response.json()

        documents = [self._parse_document(doc) for doc in result.get("documents", [])]
        total_count = result.get("documentsCount", 0)

        return documents, total_count

    async def iter_documents(
        self,
        org_id: str,
        page_size: int = 100,
        tag_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[Document, None]:
        """Iterate over all documents in an organization, automatically handling pagination.

        Args:
            org_id: The organization ID
            page_size: Number of documents to fetch per page
            tag_ids: Optional tags filter

        Yields:
            Document: Extracted document instances
        """
        page_index = 0
        while True:
            documents, _ = await self.list_documents(
                org_id=org_id,
                page_index=page_index,
                page_size=page_size,
                tag_ids=tag_ids,
            )
            if not documents:
                break
            for doc in documents:
                yield doc
            if len(documents) < page_size:
                break
            page_index += 1

    async def get_document(self, org_id: str, document_id: str) -> Document:
        """Get a document by its ID.

        Args:
            org_id: The organization ID
            document_id: The document ID

        Returns:
            Document: The document
        """
        response = await self._request(
            "GET", f"/organizations/{org_id}/documents/{document_id}"
        )
        doc = response.json()["document"]

        return self._parse_document(doc)

    async def get_document_file(self, org_id: str, document_id: str) -> bytes:
        """Get a document file content by its ID.

        Args:
            org_id: The organization ID
            document_id: The document ID

        Returns:
            bytes: The file content
        """
        response = await self._request(
            "GET", f"/organizations/{org_id}/documents/{document_id}/file"
        )
        return response.content

    async def update_document_content(
        self,
        org_id: str,
        document_id: str,
        content: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Document:
        """Update a document's name or content.

        Args:
            org_id: The organization ID
            document_id: The document ID
            content: Optional new content
            name: Optional new name

        Returns:
            Document: The updated document
        """
        data = {}
        if content is not None:
            data["content"] = content
        if name is not None:
            data["name"] = name

        if not data:
            raise ValueError("At least one of 'content' or 'name' must be provided")

        response = await self._request(
            "PATCH", f"/organizations/{org_id}/documents/{document_id}", json=data
        )
        doc = response.json()["document"]

        return self._parse_document(doc)

    async def delete_document(self, org_id: str, document_id: str) -> None:
        """Delete a document by its ID.

        Args:
            org_id: The organization ID
            document_id: The document ID
        """
        await self._request(
            "DELETE", f"/organizations/{org_id}/documents/{document_id}"
        )

    async def search_documents(
        self,
        org_id: str,
        search_query: str,
        page_index: int = 0,
        page_size: int = 100,
    ) -> tuple[List[Document], int]:
        """Search documents in the organization.

        Args:
            org_id: The organization ID
            search_query: The search query
            page_index: The page index to start from (default: 0)
            page_size: The number of documents to return (default: 100)

        Returns:
            Tuple[List[Document], int]: List of matching documents and total count
        """
        params = {
            "searchQuery": search_query,
            "pageIndex": page_index,
            "pageSize": page_size,
        }

        response = await self._request(
            "GET", f"/organizations/{org_id}/documents/search", params=params
        )
        result = response.json()

        documents = [self._parse_document(doc) for doc in result.get("documents", [])]
        total_count = result.get("totalCount", 0)

        return documents, total_count

    async def get_organization_stats(self, org_id: str) -> OrganizationStats:
        """Get statistics for the organization.

        Args:
            org_id: The organization ID

        Returns:
            OrganizationStats: The organization statistics
        """
        response = await self._request(
            "GET", f"/organizations/{org_id}/documents/statistics"
        )
        stats = response.json()["organizationStats"]

        return OrganizationStats(
            documents_count=stats["documentsCount"],
            documents_size=stats["documentsSize"],
        )

    # Tag operations
    async def list_tags(self, org_id: str) -> List[Tag]:
        """List all tags in the organization.

        Args:
            org_id: The organization ID

        Returns:
            List[Tag]: List of tags
        """
        response = await self._request("GET", f"/organizations/{org_id}/tags")
        tags_data = response.json()["tags"]

        return [
            Tag(
                id=tag["id"],
                name=tag["name"],
                color=tag["color"],
                description=tag.get("description"),
                organization_id=org_id,
            )
            for tag in tags_data
        ]

    async def create_tag(
        self,
        org_id: str,
        name: str,
        color: str,
        description: Optional[str] = None,
    ) -> Tag:
        """Create a new tag in the organization.

        Args:
            org_id: The organization ID
            name: The tag name
            color: The tag color in hex format (e.g. #000000)
            description: Optional tag description

        Returns:
            Tag: The created tag
        """
        data = {"name": name, "color": color}
        if description:
            data["description"] = description

        response = await self._request(
            "POST", f"/organizations/{org_id}/tags", json=data
        )
        tag_data = response.json()["tag"]

        return Tag(
            id=tag_data["id"],
            name=tag_data["name"],
            color=tag_data["color"],
            description=tag_data.get("description"),
            organization_id=org_id,
        )

    async def get_or_create_tag(
        self,
        org_id: str,
        name: str,
        color: str,
        description: Optional[str] = None,
    ) -> Tag:
        """Get an existing tag or create it if it doesn't exist.

        Args:
            org_id: The organization ID
            name: The tag name
            color: The tag color in hex format
            description: Optional tag description

        Returns:
            Tag: The existing or created tag
        """
        existing_tags = await self.list_tags(org_id)
        for tag in existing_tags:
            if tag.name.lower() == name.lower():
                return tag

        try:
            return await self.create_tag(org_id, name, color, description)
        except PapraClientError:
            # Recheck if tag was created concurrently
            existing_tags = await self.list_tags(org_id)
            for tag in existing_tags:
                if tag.name.lower() == name.lower():
                    return tag
            raise

    async def update_tag(
        self,
        org_id: str,
        tag_id: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tag:
        """Update a tag.

        Args:
            org_id: The organization ID
            tag_id: The tag ID
            name: Optional new name
            color: Optional new color
            description: Optional new description

        Returns:
            Tag: The updated tag
        """
        data = {}
        if name is not None:
            data["name"] = name
        if color is not None:
            data["color"] = color
        if description is not None:
            data["description"] = description

        if not data:
            raise ValueError("At least one field must be provided to update")

        response = await self._request(
            "PUT", f"/organizations/{org_id}/tags/{tag_id}", json=data
        )
        tag_data = response.json()["tag"]

        return Tag(
            id=tag_data["id"],
            name=tag_data["name"],
            color=tag_data["color"],
            description=tag_data.get("description"),
            organization_id=org_id,
        )

    async def delete_tag(self, org_id: str, tag_id: str) -> None:
        """Delete a tag by its ID.

        Args:
            org_id: The organization ID
            tag_id: The tag ID
        """
        await self._request("DELETE", f"/organizations/{org_id}/tags/{tag_id}")

    async def add_tag_to_document(
        self, org_id: str, document_id: str, tag_id: str
    ) -> None:
        """Associate a tag to a document.

        Args:
            org_id: The organization ID
            document_id: The document ID
            tag_id: The tag ID
        """
        await self._request(
            "POST",
            f"/organizations/{org_id}/documents/{document_id}/tags",
            json={"tagId": tag_id},
        )

    async def remove_tag_from_document(
        self, org_id: str, document_id: str, tag_id: str
    ) -> None:
        """Remove a tag from a document.

        Args:
            org_id: The organization ID
            document_id: The document ID
            tag_id: The tag ID
        """
        await self._request(
            "DELETE",
            f"/organizations/{org_id}/documents/{document_id}/tags/{tag_id}",
        )
