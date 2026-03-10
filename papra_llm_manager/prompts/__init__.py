"""Prompts for LLM operations."""

import os
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name.

    Args:
        name: Prompt file name without extension

    Returns:
        Prompt content as string
    """
    prompt_path = _PROMPTS_DIR / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {name}")
    return prompt_path.read_text(encoding="utf-8")


def get_extract_text_prompt(document_name: str = "") -> str:
    """Get the prompt for extracting text from images.

    Args:
        document_name: Optional document name for context

    Returns:
        Formatted prompt string
    """
    template = load_prompt("extract_text")
    return template.format(document_name=document_name)


def get_generate_tags_prompt(
    document_name: str = "",
    existing_tags: str = "none",
    max_tags: int = 5,
    text: str = "",
) -> str:
    """Get the prompt for generating tags.

    Args:
        document_name: Document name for context
        existing_tags: Comma-separated existing tags
        max_tags: Maximum number of tags to generate
        text: Document text content

    Returns:
        Formatted prompt string
    """
    template = load_prompt("generate_tags")
    return template.format(
        document_name=document_name,
        existing_tags=existing_tags,
        max_tags=max_tags,
        text=text[:4000],
    )


def get_extract_metadata_prompt(document_name: str = "", text: str = "") -> str:
    """Get the prompt for extracting metadata.

    Args:
        document_name: Document name for context
        text: Document text content

    Returns:
        Formatted prompt string
    """
    template = load_prompt("extract_metadata")
    return template.format(
        document_name=document_name,
        text=text[:4000],
    )