"""LLM abstraction layer for text extraction and tagging.

This module provides a pluggable interface for using different LLM providers
to extract text from images and generate intelligent tags for documents.
"""

import asyncio
import base64
import io
import json
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image


class LLMError(Exception):
    """Base exception for LLM handler errors."""

    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    LLM providers implement methods for extracting text from images,
    generating tags, and extracting metadata from documents.
    """

    @abstractmethod
    async def extract_text_from_image(
        self, image: Image.Image, document_name: str = ""
    ) -> str:
        """Extract text from an image/document.

        Args:
            image: PIL Image object
            document_name: Optional document name for context

        Returns:
            str: Extracted text content
        """
        pass

    @abstractmethod
    async def generate_tags(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        document_name: str = "",
        existing_tags: Optional[List[str]] = None,
        max_tags: int = 5,
    ) -> List[str]:
        """Generate relevant tags from document content.

        Args:
            text: Document text content
            image: Optional PIL Image for visual analysis
            document_name: Document name for context
            existing_tags: List of existing tags to avoid duplicates
            max_tags: Maximum number of tags to return

        Returns:
            List[str]: Generated tag names
        """
        pass

    @abstractmethod
    async def extract_metadata(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        document_name: str = "",
    ) -> dict:
        """Extract structured metadata from document.

        Args:
            text: Document text content
            image: Optional PIL Image for visual analysis
            document_name: Document name for context

        Returns:
            dict: Extracted metadata (dates, entities, amounts, etc.)
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude implementation with multimodal support.

    Uses Claude 3.5 Sonnet by default for optimal performance on vision tasks.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 8192,
    ):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model to be used (default: claude-3-5-sonnet-20241022)
            max_tokens: Maximum tokens for responses
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def extract_text_from_image(
        self, image: Image.Image, document_name: str = ""
    ) -> str:
        """Extract text from an image using Claude's vision capabilities."""
        base64_image = self._image_to_base64(image)

        message_content = [
            {
                "type": "text",
                "text": (
                    f"Extract all text from this document/image. "
                    f"Document name: {document_name}\n\n"
                    "Provide the complete, accurate text content. "
                    "Maintain the structure and formatting as much as possible. "
                    "If this is a form, invoice, receipt, or similar document, "
                    "extract all fields, dates, amounts, and line items accurately.\n\n"
                    "Return only the extracted text, no explanations or commentary."
                ),
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                },
            },
        ]

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": message_content}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise LLMError(f"Failed to extract text with Anthropic: {e}")

    async def generate_tags(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        document_name: str = "",
        existing_tags: Optional[List[str]] = None,
        max_tags: int = 5,
    ) -> List[str]:
        """Generate relevant tags using Claude."""
        existing_tags_str = ", ".join(existing_tags) if existing_tags else "none"

        prompt = f"""Generate {max_tags} relevant tags for this document.

Document name: {document_name}
Existing tags: {existing_tags_str}

Document content:
{text[:4000]}  # Truncate for context window management

Rules:
- Generate {max_tags} or fewer tags
- Use simple, clear tag names (1-3 words each)
- Avoid tags that already exist in the existing tags list
- Focus on document type, category, and key themes
- Common useful tags: invoice, receipt, contract, report, personal, work, financial, legal, tax, medical, insurance, bank, utility, statement, form, application, certificate, license, diploma, transcript, resume, portfolio, project, proposal, presentation, manual, guide, policy, procedure, memo, notice, announcement, newsletter, article, blog, research, paper, thesis, dissertation, book, chapter, essay, story, poem, script, lyrics, code, config, log, data, database, spreadsheet, chart, graph, diagram, map, plan, design, sketch, drawing, painting, photo, image, screenshot, screenshot, scan, print, copy, draft, final, version, revision, update, archive, backup, restore, import, export, sync, upload, download, share, send, receive, forward, reply, attachment, link, reference, citation, quote, note, comment, annotation, highlight, bookmark, tag, label, category, folder, directory, collection, album, playlist, channel, feed, stream, video, audio, podcast, broadcast, live, recording, transcript, subtitle, caption, translation, localization, internationalization, globalization, accessibility, compliance, security, privacy, gdpr, ccpa, hippa, pci, soc2, iso, gdpr, ccpa, hipaa, fda, sec, irs, dot, osha, epa, nrc, fcc, ftc, doj, state, local, federal, international, global, world, country, region, city, zip, address, location, place, venue, site, online, offline, cloud, local, remote, server, client, web, mobile, desktop, tablet, phone, laptop, computer, device, hardware, software, system, platform, application, app, tool, utility, service, api, sdk, library, framework, package, module, component, widget, plugin, extension, addon, theme, template, style, design, layout, ui, ux, frontend, backend, fullstack, devops, ops, admin, management, analytics, reporting, monitoring, logging, testing, quality, assurance, qa, audit, review, approval, signature, authorization, authentication, encryption, decryption, hashing, encoding, decoding, compression, decompression, optimization, performance, scalability, reliability, availability, consistency, durability, persistence, caching, storage, backup, recovery, disaster, business, continuity, strategy, planning, budgeting, forecasting, accounting, finance, economics, marketing, sales, customer, support, service, help, training, education, learning, teaching, research, development, innovation, improvement, enhancement, maintenance, troubleshooting, debugging, fixing, resolving, solving, answering, responding, communicating, collaborating, coordinating, managing, leading, directing, supervising, overseeing, controlling, regulating, governing, administering, operating, executing, implementing, deploying, releasing, publishing, distributing, marketing, selling, buying, purchasing, ordering, shipping, delivering, receiving, accepting, rejecting, returning, refunding, exchanging, repairing, maintaining, servicing, updating, upgrading, renewing, cancelling, terminating, closing, opening, starting, stopping, pausing, resuming, restarting, reloading, refreshing, resetting, clearing, deleting, removing, archiving, restoring, recovering, backing, syncing, uploading, downloading, sharing, sending, receiving, forwarding, replying

Return the tags as a JSON array of strings, like:
["tag1", "tag2", "tag3"]
"""

        content = [{"type": "text", "text": prompt}]

        if image:
            base64_image = self._image_to_base64(image)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                }
            )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": content}],
            )

            result = response.content[0].text.strip()
            # Try to extract JSON from the response
            if "[" in result and "]" in result:
                start = result.index("[")
                end = result.rindex("]") + 1
                result = result[start:end]
            tags = json.loads(result)
            return tags[:max_tags]
        except json.JSONDecodeError:
            # Fallback: extract tags from text response
            return []
        except Exception as e:
            raise LLMError(f"Failed to generate tags with Anthropic: {e}")

    async def extract_metadata(
        self, text: str, image: Optional[Image.Image] = None, document_name: str = ""
    ) -> dict:
        """Extract structured metadata using Claude."""
        prompt = f"""Extract structured metadata from this document.

Document name: {document_name}

Document content:
{text[:4000]}

Extract the following metadata fields if present:
- title: Document title or subject
- date: Any dates found (ISO format YYYY-MM-DD preferred)
- amount: Any monetary amounts or numbers
- company: Company or organization names
- person: Person names
- location: Locations, addresses, places
- phone: Phone numbers
- email: Email addresses
- document_type: Type of document (invoice, receipt, contract, etc.)
- status: Status if applicable (draft, final, paid, pending, etc.)
- reference: Reference numbers, IDs, invoice numbers, etc.

Return as JSON object. Use null for missing fields:
{{
  "title": null,
  "date": null,
  "amount": null,
  "company": null,
  "person": null,
  "location": null,
  "phone": null,
  "email": null,
  "document_type": null,
  "status": null,
  "reference": null
}}
"""

        content = [{"type": "text", "text": prompt}]

        if image:
            base64_image = self._image_to_base64(image)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                }
            )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": content}],
            )

            result = response.content[0].text.strip()
            if "{" in result and "}" in result:
                start = result.index("{")
                end = result.rindex("}") + 1
                result = result[start:end]
            metadata = json.loads(result)
            return metadata
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            raise LLMError(f"Failed to extract metadata with Anthropic: {e}")


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4 Vision implementation.

    Uses GPT-4o by default for optimal multimodal capabilities.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
    ):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model to be used (default: gpt-4o)
            max_tokens: Maximum tokens for responses
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def extract_text_from_image(
        self, image: Image.Image, document_name: str = ""
    ) -> str:
        """Extract text from an image using GPT-4 Vision."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Extract all text from this document. "
                                    f"Document name: {document_name}\n\n"
                                    "Provide the complete, accurate text content. "
                                    "Maintain structure and formatting. "
                                    "If this is a form, invoice, receipt, or similar document, "
                                    "extract all fields, dates, amounts, and line items accurately.\n\n"
                                    "Return only the extracted text, no explanations."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"Failed to extract text with OpenAI: {e}")

    async def generate_tags(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        document_name: str = "",
        existing_tags: Optional[List[str]] = None,
        max_tags: int = 5,
    ) -> List[str]:
        """Generate relevant tags using GPT-4."""
        existing_tags_str = ", ".join(existing_tags) if existing_tags else "none"

        prompt = f"""Generate {max_tags} relevant tags for this document.

Document name: {document_name}
Existing tags: {existing_tags_str}

Document content:
{text[:4000]}

Generate {max_tags} or fewer tags. Use simple, clear names. Avoid existing tags.
Focus on document type, category, and key themes.

Return as JSON array: ["tag1", "tag2", "tag3"]
"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                    },
                }
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
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
            raise LLMError(f"Failed to generate tags with OpenAI: {e}")

    async def extract_metadata(
        self, text: str, image: Optional[Image.Image] = None, document_name: str = ""
    ) -> dict:
        """Extract structured metadata using GPT-4."""
        prompt = f"""Extract structured metadata from this document.

Document name: {document_name}
Content: {text[:4000]}

Extract: title, date, amount, company, person, location, phone, email, document_type, status, reference.
Return as JSON object with null for missing fields.
"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                    },
                }
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=1000
            )

            result = response.choices[0].message.content.strip()
            if "{" in result and "}" in result:
                start = result.index("{")
                end = result.rindex("}") + 1
                result = result[start:end]
            return json.loads(result)
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            raise LLMError(f"Failed to extract metadata with OpenAI: {e}")


class DeepSeekProvider(LLMProvider):
    """DeepSeek implementation.

    Uses DeepSeek API for text requests (api.deepseek.com)
    Uses SiliconFlow API for vision requests (api.siliconflow.cn/v1, OpenAI-compatible)
    Default text model: deepseek-chat
    Default vision model: zai-org/GLM-4.6V (from SiliconFlow)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        vision_model: str = "zai-org/GLM-4.6V",
        max_tokens: int = 8192,
    ):
        """Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek/SiliconFlow API key
            model: Model to use for text requests (default: deepseek-chat)
            vision_model: Model to use for vision requests (default: zai-org/GLM-4.6V)
            max_tokens: Maximum tokens for responses
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        # Text client uses DeepSeek API
        self.text_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        # Vision client uses SiliconFlow API (OpenAI-compatible)
        self.vision_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
        )

        self.model = model
        self.vision_model = vision_model
        self.max_tokens = max_tokens

    async def extract_text_from_image(
        self, image: Image.Image, document_name: str = ""
    ) -> str:
        """Extract text from an image using SiliconFlow's vision capabilities."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        try:
            response = await self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Extract all text from this document. "
                                    f"Document name: {document_name}\n\n"
                                    "Provide the complete, accurate text content. "
                                    "Maintain structure and formatting. "
                                    "If this is a form, invoice, receipt, or similar document, "
                                    "extract all fields, dates, amounts, and line items accurately.\n\n"
                                    "Return only the best extracted text, no explanations."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"Failed to extract text with DeepSeek/SiliconFlow: {e}")

    async def generate_tags(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        document_name: str = "",
        existing_tags: Optional[List[str]] = None,
        max_tags: int = 5
    ) -> List[str]:
        """Generate relevant tags using DeepSeek (text only - no image support for tagging)."""
        existing_tags_str = ", ".join(existing_tags) if existing_tags else "none"

        prompt = f"""Generate {max_tags} relevant tags for this document.

Document name: {document_name}
Existing tags: {existing_tags_str}

Document content:
{text[:4000]}

Generate {max_tags} or fewer tags. Use simple, clear names. Avoid existing tags.
Focus on document type, category, and key themes.

Return as JSON array: ["tag1", "tag2", "tag3"]
"""

        try:
            response = await self.text_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
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
            raise LLMError(f"Failed to generate tags with DeepSeek: {e}")

    async def extract_metadata(
        self, text: str, image: Optional[Image.Image] = None, document_name: str = ""
    ) -> dict:
        """Extract structured metadata using DeepSeek (text only)."""
        prompt = f"""Extract structured metadata from this document.

Document name: {document_name}
Content: {text[:4000]}

Extract: title, date, amount, company, person, location, phone, email, document_type, status, reference.
Return as JSON object with null for missing fields.
"""

        try:
            response = await self.text_client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=1000
            )

            result = response.choices[0].message.content.strip()
            if "{" in result and "}" in result:
                start = result.index("{")
                end = result.rindex("}") + 1
                result = result[start:end]
            return json.loads(result)
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            raise LLMError(f"Failed to extract metadata with DeepSeek: {e}")


def create_llm_provider(
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    max_tokens: int = 8192,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider: Provider name ("anthropic", "openai", or "deepseek")
        api_key: API key for the provider
        model: Optional model name (uses default if not specified)
        max_tokens: Maximum tokens for responses

    Returns:
        LLMProvider: Configured provider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    provider = provider.lower()

    if provider == "anthropic":
        if model is None:
            model = "claude-3-5-sonnet-20241022"
        return AnthropicProvider(api_key=api_key, model=model, max_tokens=max_tokens)
    elif provider == "openai":
        if model is None:
            model = "gpt-4o"
        return OpenAIProvider(api_key=api_key, model=model, max_tokens=max_tokens)
    elif provider == "deepseek":
        if model is None:
            model = "deepseek-chat"
        return DeepSeekProvider(api_key=api_key, model=model, max_tokens=max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Use 'anthropic', 'openai', or 'deepseek'"
        )
