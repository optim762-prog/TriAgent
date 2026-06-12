from typing import Literal, Dict, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict

AblationMode = Literal["full", "no_knowledge", "no_execution"]
Provider = Literal["openai", "groq", "deepseek", "llamacpp_a", "llamacpp_b"]

# Cost table: model_name -> (input_usd_per_1M_tokens, output_usd_per_1M_tokens)
# Local models cost 0. Update as pricing changes.
DEFAULT_COST_TABLE: Dict[str, Tuple[float, float]] = {
    "gpt-4o":                    (2.50,  10.00),
    "gpt-4o-mini":               (0.15,   0.60),
    "llama-3.3-70b-versatile":   (0.59,   0.79),
    "deepseek-chat":             (0.27,   1.10),
    "gemma4-local":              (0.0,    0.0),
    "qwen-local":                (0.0,    0.0),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRIAGENT_",
        extra="ignore",
        case_sensitive=False,
    )

    # API keys (no prefix — read directly from env)
    openai_api_key: str = ""
    groq_api_key: str = ""
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""
    openweather_api_key: str = ""

    # Backbone selection
    backbone: str = "gpt-4o"
    backbone_provider: Provider = "openai"

    # llama.cpp server endpoints
    llamacpp_a_base_url: str = "http://machine-a:8080/v1"
    llamacpp_b_base_url: str = "http://localhost:8080/v1"

    # Ablation
    ablation_mode: AblationMode = "full"

    # Runtime behaviour
    confidence_threshold: float = 0.6
    max_tool_iterations: int = 10
    rules_path: str = "data/rules"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def cost_table(self) -> Dict[str, Tuple[float, float]]:
        return DEFAULT_COST_TABLE

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
