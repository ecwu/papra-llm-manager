"""Tests for papra-llm-manager models."""

import pytest
from datetime import datetime
from papra_llm_manager.models import Tag, Document, ProcessingResult, Organization


def test_tag_creation():
    tag = Tag(id="tag1", name="invoice", color="#FF0000")
    assert tag.id == "tag1"
    assert tag.name == "invoice"
    assert tag.color == "#FF0000"
    assert tag.description is None


def test_tag_color_validation_valid():
    tag = Tag(id="tag1", name="test", color="#FF0000")
    assert tag.color == "#FF0000"


def test_tag_color_validation_short():
    tag = Tag(id="tag1", name="test", color="#F00")
    assert tag.color == "#F00"


def test_tag_color_validation_invalid():
    with pytest.raises(ValueError):
        Tag(id="tag1", name="test", color="FF0000")


def test_document_creation():
    doc = Document(
        id="doc1",
        name="test.pdf",
        organization_id="org1",
        content="Test content",
        size=1024,
        mime_type="application/pdf",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    assert doc.id == "doc1"
    assert doc.has_text is True
    assert doc.text_length == 12


def test_document_has_text_false():
    doc = Document(
        id="doc1",
        name="test.pdf",
        organization_id="org1",
        content="   ",
        size=1024,
        mime_type="application/pdf",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    assert doc.has_text is False


def test_document_size_validation():
    with pytest.raises(ValueError):
        Document(
            id="doc1",
            name="test.pdf",
            organization_id="org1",
            content="Test",
            size=-1,
            mime_type="application/pdf",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )


def test_processing_result_creation():
    result = ProcessingResult(success=True)
    assert result.success is True
    assert result.text_extracted is False
    assert len(result.tags_added) == 0


def test_organization_stats_validation():
    from papra_llm_manager.models import OrganizationStats

    stats = OrganizationStats(documents_count=100, documents_size=1024000)
    assert stats.documents_count == 100
    assert stats.documents_size == 1024000


def test_organization_stats_invalid_count():
    from papra_llm_manager.models import OrganizationStats

    with pytest.raises(ValueError):
        OrganizationStats(documents_count=-1, documents_size=1024000)
