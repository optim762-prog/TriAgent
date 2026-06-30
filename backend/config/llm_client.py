from openai import OpenAI
from .settings import settings, Provider


def make_client(provider: Provider = None) -> OpenAI:
    """
    Returns an OpenAI-SDK client pointed at the right endpoint for each provider.
    DeepSeek and llama.cpp both expose OpenAI-compatible /v1 endpoints,
    so the same SDK handles all three paper backbones.
    """
    p = provider or settings.backbone_provider

    if p == "deepseek":
        return OpenAI(
            api_key=settings.deepseek_api_key or "placeholder",
            base_url="https://api.deepseek.com/v1",
        )
    if p == "llamacpp_b":
        return OpenAI(
            api_key="lcpp",
            base_url=settings.llamacpp_b_base_url,
        )
    raise ValueError(f"Unknown provider: {p!r}. Use 'deepseek' or 'llamacpp_b'.")
