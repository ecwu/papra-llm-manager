"""papra-llm-manager - Enhance Papra with AI-powered features.

This package provides tools for:
- Uploading documents to Papra
- Extracting text from images using LLM (Claude, GPT-4 Vision)
- Intelligent tagging based on document understanding
- Batch processing of documents
"""

from papra_llm_manager.client import (
    PapraClient,
    PapraClientError,
    PapraAuthenticationError,
    PapraNotFoundError,
)
from papra_llm_manager.config import Config
from papra_llm_manager.exceptions import (
    ProcessingError,
    TaggingError,
    TextExtractionError,
    LLMProviderError,
    ValidationError,
)
from papra_llm_manager.llm_handler import (
    LLMProvider,
    LLMError,
    AnthropicProvider,
    OpenAIProvider,
    DeepSeekProvider,
    create_llm_provider,
)
from papra_llm_manager.models import (
    Document,
    Tag,
    Organization,
    OrganizationStats,
    ProcessingResult,
    TagRule,
    ApiKey,
)
from papra_llm_manager.processors import DocumentProcessor
from papra_llm_manager.tagger import DocumentTagger
from papra_llm_manager.services import (
    PapraServiceFactory,
    create_services_from_config,
)

__version__ = "0.1.0"
__all__ = [
    # Client
    "PapraClient",
    "PapraClientError",
    "PapraAuthenticationError",
    "PapraNotFoundError",
    # Config
    "Config",
    # Exceptions
    "ProcessingError",
    "TaggingError",
    "TextExtractionError",
    "LLMProviderError",
    "ValidationError",
    # LLM Handler
    "LLMProvider",
    "LLMError",
    "AnthropicProvider",
    "OpenAIProvider",
    "create_llm_provider",
    # Models
    "Document",
    "Tag",
    "Organization",
    "OrganizationStats",
    "ProcessingResult",
    "TagRule",
    "ApiKey",
    # Processors
    "DocumentProcessor",
    "DocumentTagger",
    # Services
    "PapraServiceFactory",
    "create_services_from_config",
]
