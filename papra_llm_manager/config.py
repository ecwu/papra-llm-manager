"""Configuration management for papra-llm-manager."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for papra-llm-manager."""

    # Papra config
    papra_api_token: str
    papra_org_id: str
    papra_base_url: str = "https://vercel.ai/api"

    # LLM config
    llm_provider: str = "deepseek"  # or "anthropic", "openai", "deepseek"
    llm_api_key: str = ""
    llm_model: str = ""

    # Processing config
    batch_size: int = 10
    max_tags: int = 5
    extract_text_threshold: int = 100  # chars below which to re-extract
    default_tag_color: str = "#3B82F6"
    tag_colors: dict = field(default_factory=lambda: {
        "invoice": "#EF4444",
        "receipt": "#F59E0B",
        "contract": "#3B82F6",
        "report": "#10B981",
        "personal": "#8B5CF6",
        "work": "#EC4899",
        "financial": "#F97316",
        "legal": "#6366F1",
        "medical": "#14B8A6",
        "tax": "#DC2626",
    })

    # Retry config
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Config":
        """Load configuration from environment variables.

        Args:
            env_file: Path to .env file

        Returns:
            Config instance
        """
        # Load environment variables from .env file if it exists
        load_dotenv(env_file)

        papra_api_token = os.getenv("PAPRA_API_TOKEN", "")
        papra_org_id = os.getenv("PAPRA_ORG_ID", "")

        if not papra_api_token:
            raise ValueError("PAPRA_API_TOKEN environment variable is required")
        if not papra_org_id:
            raise ValueError("PAPRA_ORG_ID environment variable is required")

        # Get LLM config
        llm_provider = os.getenv("LLM_PROVIDER", "deepseek")
        llm_api_key = os.getenv("LLM_API_KEY", "")
        llm_model = os.getenv("LLM_MODEL", "")

        # Handle DeepSeek provider with specific env vars
        if llm_provider == "deepseek":
            if not llm_api_key:
                # Try DEEPSEEK_API_KEY or SILICONFLOW_API_KEY
                llm_api_key = os.getenv("DEEPSEEK_API_KEY", "")
                if not llm_api_key:
                    llm_api_key = os.getenv("SILICONFLOW_API_KEY", "")

        return cls(
            papra_api_token=papra_api_token,
            papra_org_id=papra_org_id,
            papra_base_url=os.getenv("PAPRA_BASE_URL", "https://vercel.ai/api"),
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_tags=int(os.getenv("MAX_TAGS", "5")),
            extract_text_threshold=int(os.getenv("EXTRACT_TEXT_THRESHOLD", "100")),
            default_tag_color=os.getenv("DEFAULT_TAG_COLOR", "#3B82F6"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
        )

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from a YAML or TOML file.

        Args:
            path: Path to config file (.yaml, .yml, or .toml)

        Returns:
            Config instance
        """
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if config_path.suffix in {".yaml", ".yml"}:
            import yaml

            with open(config_path) as f:
                data = yaml.safe_load(f)
        elif config_path.suffix == ".toml":
            import tomli

            with open(config_path, "rb") as f:
                data = tomli.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls(**data)

    def get_tag_color(self, tag_name: str) -> str:
        """Get color hex for a tag name.

        Args:
            tag_name: Name of the tag

        Returns:
            Hex color code
        """
        tag_lower = tag_name.lower()
        return self.tag_colors.get(tag_lower, self.default_tag_color)
