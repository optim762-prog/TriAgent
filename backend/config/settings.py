from typing import Literal, Dict, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict

AblationMode = Literal["full", "no_knowledge", "no_execution"]
Provider = Literal["deepseek", "llamacpp_b"]

# Cost table: model_name -> (input_usd_per_1M_tokens, output_usd_per_1M_tokens)
DEFAULT_COST_TABLE: Dict[str, Tuple[float, float]] = {
    "deepseek-chat":  (0.27,  1.10),
    "gemma4-local":   (0.0,   0.0),
    "qwen-local":     (0.0,   0.0),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    # API keys (no prefix — read directly from env)
    deepseek_api_key: str = ""
    openweather_api_key: str = ""

    # Backbone selection
    backbone: str = "deepseek-chat"
    backbone_provider: Provider = "deepseek"

    # llama.cpp server endpoint (GPU machine)
    llamacpp_b_base_url: str = "http://localhost:8080/v1"

    # Ablation
    ablation_mode: AblationMode = "full"

    # Runtime behaviour
    confidence_threshold: float = 0.6
    max_tool_iterations: int = 10
    rules_path: str = "data/rules"

    @property
    def cost_table(self) -> Dict[str, Tuple[float, float]]:
        return DEFAULT_COST_TABLE


settings = Settings()
