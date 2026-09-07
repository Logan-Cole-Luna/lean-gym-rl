from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class APIKeys(BaseModel):
    """
    Credentials for external model providers.

    SecretStr prevents API keys from being printed accidentally.
    """

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    huggingface_token: SecretStr | None = None


class HuggingFaceEnvironment(BaseModel):
    """
    Configuration for running open-source Hugging Face models.
    """

    device: str = "auto"

    dtype: Literal[
        "auto",
        "float32",
        "float16",
        "bfloat16",
    ] = "auto"

    cache_dir: Path | None = None

    trust_remote_code: bool = False

    # Optional quantization settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Useful for large models
    device_map: str | dict | None = "auto"

    # Maximum sequence length used by the judge
    max_model_length: int | None = None


class GenerationConfig(BaseModel):
    """
    Parameters used when asking the judge model for a judgment.
    """

    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0


class JudgeConfig(BaseModel):
    model_name: str
    provider: ModelProvider

    api_keys: APIKeys = Field(default_factory=APIKeys)

    generation: GenerationConfig = Field(
        default_factory=GenerationConfig
    )

    huggingface_environment: HuggingFaceEnvironment | None = None