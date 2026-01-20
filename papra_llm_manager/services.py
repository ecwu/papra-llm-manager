"""Service factory for creating configured services."""

from typing import TYPE_CHECKING, Optional

from papra_llm_manager.config import Config
from papra_llm_manager.client import PapraClient
from papra_llm_manager.llm_handler import LLMProvider, create_llm_provider
from papra_llm_manager.processors import DocumentProcessor

if TYPE_CHECKING:
    from papra_llm_manager.tagger import DocumentTagger


class PapraServiceFactory:
    """Factory for creating configured Papra services."""

    @staticmethod
    def create_client(config: Config) -> PapraClient:
        """Create a Papra client from configuration.

        Args:
            config: Configuration object

        Returns:
            PapraClient: Configured client
        """
        return PapraClient(
            api_token=config.papra_api_token,
            base_url=config.papra_base_url,
        )

    @staticmethod
    def create_llm_handler(config: Config) -> LLMProvider:
        """Create an LLM handler from configuration.

        Args:
            config: Configuration object

        Returns:
            LLMProvider: Configured LLM handler
        """
        return create_llm_provider(
            provider=config.llm_provider,
            api_key=config.llm_api_key,
            model=config.llm_model if config.llm_model else None,
            max_tokens=8192,
        )

    @staticmethod
    def create_processor(
        config: Config,
        papra_client: Optional[PapraClient] = None,
        llm_handler: Optional[LLMProvider] = None,
        tagger: Optional["DocumentTagger"] = None,
    ) -> DocumentProcessor:
        """Create a document processor from configuration.

        Args:
            config: Configuration object
            papra_client: Optional pre-configured client
            llm_handler: Optional pre-configured LLM handler
            tagger: Optional pre-configured tagger

        Returns:
            DocumentProcessor: Configured processor
        """
        client = papra_client or PapraServiceFactory.create_client(config)
        llm = llm_handler or PapraServiceFactory.create_llm_handler(config)

        return DocumentProcessor(
            papra_client=client,
            llm_handler=llm,
            extract_text_threshold=config.extract_text_threshold,
            max_tags=config.max_tags,
            tagger=tagger,
        )

    @staticmethod
    def create_tagger(
        config: Config,
        papra_client: Optional[PapraClient] = None,
        llm_handler: Optional[LLMProvider] = None,
    ) -> "DocumentTagger":
        """Create a document tagger from configuration.

        Args:
            config: Configuration object
            papra_client: Optional pre-configured client
            llm_handler: Optional pre-configured LLM handler

        Returns:
            DocumentTagger: Configured tagger
        """
        from papra_llm_manager.tagger import DocumentTagger

        client = papra_client or PapraServiceFactory.create_client(config)
        llm = llm_handler or PapraServiceFactory.create_llm_handler(config)

        return DocumentTagger(
            papra_client=client,
            llm_handler=llm,
            tag_colors=config.tag_colors,
        )


def create_services_from_config(config: Config) -> dict:
    """Create all services from configuration.

    Args:
        config: Configuration object

    Returns:
        dict: Dictionary containing all services
    """
    client = PapraServiceFactory.create_client(config)
    llm = PapraServiceFactory.create_llm_handler(config)
    tagger = PapraServiceFactory.create_tagger(config, client, llm)
    processor = PapraServiceFactory.create_processor(config, client, llm, tagger)

    return {
        "client": client,
        "llm_handler": llm,
        "tagger": tagger,
        "processor": processor,
    }
