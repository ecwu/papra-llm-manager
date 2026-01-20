"""Test configuration for pytest."""

import pytest
from datetime import datetime
from papra_llm_manager.models import Tag, Document


@pytest.fixture
def sample_tag():
    return Tag(
        id="tag1",
        name="invoice",
        color="#FF0000",
        description="Invoice documents"
    )


@pytest.fixture
def sample_document():
    return Document(
        id="doc1",
        name="invoice.pdf",
        organization_id="org1",
        content="Invoice #12345 for $100",
        size=1024,
        mime_type="application/pdf",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        tags=[]
    )
