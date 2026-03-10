"""Tests for papra-llm-manager service factory."""

from unittest.mock import Mock, patch
import pytest
from papra_llm_manager.services import PapraServiceFactory, create_services_from_config
from papra_llm_manager.config import Config


@pytest.fixture
def mock_config():
    config = Mock(spec=Config)
    config.papra_api_token = "test_token"
    config.papra_base_url = "https://api.test.com"
    config.llm_model = "anthropic/claude-3-5-sonnet-20241022"
    config.llm_api_key = "llm_key"
    config.llm_api_base = None
    config.extract_text_threshold = 100
    config.max_tags = 5
    config.tag_colors = {}
    return config


def test_create_client(mock_config):
    client = PapraServiceFactory.create_client(mock_config)
    assert client.api_token == "test_token"
    assert client.base_url == "https://api.test.com"


def test_create_llm_handler(mock_config):
    with patch('papra_llm_manager.services.LiteLLMProvider') as MockLiteLLM:
        mock_llm = Mock()
        MockLiteLLM.return_value = mock_llm

        llm = PapraServiceFactory.create_llm_handler(mock_config)

        MockLiteLLM.assert_called_once_with(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key="llm_key",
            api_base=None,
            max_tokens=8192,
        )
        assert llm == mock_llm


def test_create_processor_with_dependencies(mock_config):
    with patch.object(PapraServiceFactory, 'create_client') as mock_create_client, \
         patch.object(PapraServiceFactory, 'create_llm_handler') as mock_create_llm:
        mock_client = Mock()
        mock_llm = Mock()
        mock_create_client.return_value = mock_client
        mock_create_llm.return_value = mock_llm

        processor = PapraServiceFactory.create_processor(mock_config)

        assert processor.papra == mock_client
        assert processor.llm == mock_llm


def test_create_processor_with_custom_dependencies(mock_config):
    mock_client = Mock()
    mock_llm = Mock()
    mock_tagger = Mock()

    processor = PapraServiceFactory.create_processor(
        mock_config,
        papra_client=mock_client,
        llm_handler=mock_llm,
        tagger=mock_tagger
    )

    assert processor.papra == mock_client
    assert processor.llm == mock_llm
    assert processor.tagger == mock_tagger


def test_create_services_from_config(mock_config):
    with patch.object(PapraServiceFactory, 'create_client') as mock_create_client, \
         patch.object(PapraServiceFactory, 'create_llm_handler') as mock_create_llm:
        mock_client = Mock()
        mock_llm = Mock()
        mock_create_client.return_value = mock_client
        mock_create_llm.return_value = mock_llm

        services = create_services_from_config(mock_config)

        assert "client" in services
        assert "llm_handler" in services
        assert "tagger" in services
        assert "processor" in services
        assert services["client"] == mock_client
        assert services["llm_handler"] == mock_llm
