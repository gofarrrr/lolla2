from pydantic import BaseSettings, Field


class GammaConfig(BaseSettings):
    """Gamma API Configuration"""

    api_key: str = Field(..., env="GAMMA_API_KEY")
    api_base_url: str = Field(
        default="https://public-api.gamma.app/v0.2", env="GAMMA_API_URL"
    )

    # Rate limiting
    max_generations_per_month: int = 50
    rate_limit_delay: float = 2.0  # seconds between requests
    max_requests_per_minute: int = 10

    # Default presentation settings
    default_theme: str = "Oasis"
    default_format: str = "presentation"
    default_num_cards: int = 10
    default_language: str = "en"

    # Image generation settings
    default_image_source: str = "aiGenerated"
    default_image_model: str = "imagen-4-pro"
    default_image_style: str = "professional, modern, clean"

    # Timeout settings
    request_timeout: float = 30.0
    generation_timeout: float = 300.0  # 5 minutes for complex presentations

    # Storage settings
    storage_dir: str = Field(default="data/presentations", env="GAMMA_STORAGE_DIR")
    cleanup_days: int = 30  # Days to keep presentations before cleanup

    # Enterprise features
    workspace_access: str = "view"
    external_access: str = "noAccess"

    class Config:
        env_file = ".env"
        env_prefix = "GAMMA_"
