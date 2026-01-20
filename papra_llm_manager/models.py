"""Data models for Papra resources."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Tag(BaseModel):
    """Represents a tag in Papra."""

    id: str
    name: str
    color: str
    description: Optional[str] = None
    organization_id: Optional[str] = None

    @field_validator("color")
    @classmethod
    def validate_hex_color(cls, v: str) -> str:
        """Validate color is a valid hex color code."""
        if not v.startswith("#") or len(v) not in [4, 7]:
            raise ValueError("Color must be a valid hex color code (e.g., #FF0000)")
        return v


class Document(BaseModel):
    """Represents a document in Papra."""

    id: str
    name: str
    organization_id: str
    content: str
    size: int = Field(ge=0)
    mime_type: str
    created_at: datetime
    updated_at: datetime
    tags: List[Tag] = Field(default_factory=list)
    file_hash: Optional[str] = None

    @property
    def has_text(self) -> bool:
        """Check if document has meaningful text content."""
        return bool(self.content and self.content.strip())

    @property
    def text_length(self) -> int:
        """Get the length of the document's text content."""
        return len(self.content.strip()) if self.content else 0


class Organization(BaseModel):
    """Represents an organization in Papra."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime


class ProcessingResult(BaseModel):
    """Result of processing a document."""

    document: Optional[Document] = None
    success: bool
    text_extracted: bool = False
    tags_added: List[Tag] = Field(default_factory=list)
    error: Optional[str] = None
    metadata_extracted: dict = Field(default_factory=dict)

    @property
    def tags_added_names(self) -> List[str]:
        """Get names of tags added."""
        return [tag.name for tag in self.tags_added]


class TagRule(BaseModel):
    """Represents a tagging rule."""

    name: str
    tag_name: str
    condition: str  # Simple condition description or regex pattern


class ApiKey(BaseModel):
    """Represents an API key."""

    id: str
    name: str
    permissions: List[str]


class OrganizationStats(BaseModel):
    """Statistics for an organization."""

    documents_count: int = Field(ge=0)
    documents_size: int = Field(ge=0)
