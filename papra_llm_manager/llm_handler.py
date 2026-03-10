"""LLM abstraction layer using LiteLLM.

This module provides a unified interface for using different LLM providers
via LiteLLM to extract text from images and generate intelligent tags.
"""

import asyncio
import base64
import io
import json
from typing import List, Optional

from PIL import Image

from papra_llm_manager.prompts import (
    get_extract_metadata_prompt,
    get_extract_text_prompt,
    get_generate_tags_prompt,
)


class LLMError(Exception):
    """Base exception for LLM handler errors."""

    pass


class LiteLLMProvider:
    """LiteLLM implementation with unified interface.

    Supports 100+ LLM providers via LiteLLM. Uses model format: provider/model
    e.g., "ollama/gpt-oss:20b", "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        api_base: str = "",
        max_tokens: int = 8192,
    ):
        """Initialize the LiteLLM provider.

        Args:
            model: Model name with provider prefix (e.g., "ollama/gpt-oss:20b")
            api_key: API key for the provider (optional for some providers like Ollama)
            api_base: Base URL for the API (e.g., "http://localhost:11434" for Ollama)
            max_tokens: Maximum tokens for responses
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm package is required. Install with: pip install litellm"
            )

        self.model = model
        self.api_key = api_key or "fake-key"  # Ollama doesn't require real key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self._litellm = litellm

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_messages(
        self,
        text: str,
        image: Optional[Image.Image] = None,
    ) -> List[dict]:
        """Build message content list with optional image."""
        messages = []
        if text:
            messages.append({"role": "user", "content": text})

        if image:
            base64_image = self._image_to_base64(image)
            image_url = f"data:image/png;base64,{base64_image}"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text} if text else {"type": "text", "text": " "},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            )
            return messages

        return messages

    async def extract_text_from_image(
        self, image: Image.Image, document_name: str = ""
    ) -> str:
        """Extract text from an image using vision capabilities."""
        prompt = get_extract_text_prompt(document_name)
        base64_image = self._image_to_base64(image)
        image_url = f"data:image/png;base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        try:
            response = await self._litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"Failed to extract text with LiteLLM: {e}")

    async def generate_tags(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        document_name: str = "",
        existing_tags: Optional[List[str]] = None,
        max_tags: int = 5,
    ) -> List[str]:
        """Generate relevant tags from document content."""
        existing_tags_str = ", ".join(existing_tags) if existing_tags else "none"
        prompt = get_generate_tags_prompt(
            document_name=document_name,
            existing_tags=existing_tags_str,
            max_tags=max_tags,
            text=text,
        )

        content = [{"type": "text", "text": prompt}]

        if image:
            base64_image = self._image_to_base64(image)
            image_url = f"data:image/png;base64,{base64_image}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages = [{"role": "user", "content": content}]

        try:
            response = await self._litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=500,
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None,
            )

            result = response.choices[0].message.content.strip()
            if "[" in result and "]" in result:
                start = result.index("[")
                end = result.rindex("]") + 1
                result = result[start:end]
            tags = json.loads(result)
            return tags[:max_tags]
        except json.JSONDecodeError:
            return []
        except Exception as e:
            raise LLMError(f"Failed to generate tags with LiteLLM: {e}")

    async def extract_metadata(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        document_name: str = "",
    ) -> dict:
        """Extract structured metadata from document."""
        prompt = get_extract_metadata_prompt(document_name=document_name, text=text)

        content = [{"type": "text", "text": prompt}]

        if image:
            base64_image = self._image_to_base64(image)
            image_url = f"data:image/png;base64,{base64_image}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages = [{"role": "user", "content": content}]

        try:
            response = await self._litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None,
            )

            result = response.choices[0].message.content.strip()
            if "{" in result and "}" in result:
                start = result.index("{")
                end = result.rindex("}") + 1
                result = result[start:end]
            metadata = json.loads(result)
            return metadata
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            raise LLMError(f"Failed to extract metadata with LiteLLM: {e}")


LLMProvider = LiteLLMProvider