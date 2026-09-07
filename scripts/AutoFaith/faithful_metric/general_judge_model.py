from __future__ import annotations

import json
import re
import sys

from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from judge_config import JudgeConfig, ModelProvider
else:
    from .judge_config import JudgeConfig, ModelProvider


T = TypeVar("T", bound=BaseModel)


class GeneralJudgeModel:

    def __init__(self, config: JudgeConfig):
        self.config = config
        self._hf_model = None
        self._hf_tokenizer = None

    # ========================================================
    # General structured generation
    # ========================================================

    def run_structured(self, prompt: str, output_model: type[T], schema_name: str) -> T:
        match self.config.provider:
            case ModelProvider.OPENAI:
                return self._run_openai(prompt, output_model, schema_name)
            case ModelProvider.ANTHROPIC:
                return self._run_anthropic(prompt, output_model, schema_name)
            case ModelProvider.DEEPSEEK:
                return self._run_deepseek(prompt, output_model, schema_name)
            case ModelProvider.HUGGINGFACE | ModelProvider.LOCAL:
                return self._run_huggingface(prompt, output_model)
            case _:
                raise ValueError(f"Unsupported model provider: {self.config.provider}")

    def _get_key(self, key: Any, name: str) -> str:
        if key is None:
            raise ValueError(f"{name} is not configured.")
        return key.get_secret_value()

    # ========================================================
    # OpenAI
    # ========================================================

    def _run_openai(self, prompt: str, output_model: type[T], schema_name: str) -> T:
        api_key = self._get_key(self.config.api_keys.openai_api_key, "OpenAI API key")
        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=self.config.model_name,
            input=prompt,
            max_output_tokens=self.config.generation.max_tokens,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": output_model.model_json_schema(),
                    "strict": True,
                }
            },
        )

        return output_model.model_validate_json(response.output_text)

    # ========================================================
    # Anthropic
    # ========================================================

    def _run_anthropic(self, prompt: str, output_model: type[T], schema_name: str) -> T:
        import anthropic

        api_key = self._get_key(self.config.api_keys.anthropic_api_key, "Anthropic API key")
        client = anthropic.Anthropic(api_key=api_key)

        tool = {
            "name": schema_name,
            "description": "Return the requested structured output.",
            "input_schema": output_model.model_json_schema(),
        }

        response = client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.generation.max_tokens,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": schema_name},
        )

        for content in response.content:
            if content.type == "tool_use" and content.name == schema_name:
                return output_model.model_validate(content.input)

        raise RuntimeError(f"Anthropic model did not return {schema_name}.")

    # ========================================================
    # DeepSeek
    # ========================================================

    def _run_deepseek(self, prompt: str, output_model: type[T], schema_name: str) -> T:
        api_key = self._get_key(self.config.api_keys.deepseek_api_key, "DeepSeek API key")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.responses.create(
            model=self.config.model_name,
            input=prompt,
            max_output_tokens=self.config.generation.max_tokens,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": output_model.model_json_schema(),
                }
            },
        )

        return output_model.model_validate_json(response.output_text)

    # ========================================================
    # Hugging Face / Local
    # ========================================================

    def _load_huggingface_model(self) -> None:
        import torch

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        env = self.config.huggingface_environment

        if env is None:
            raise ValueError("huggingface_environment must be provided for Hugging Face or local models.")

        if env.load_in_4bit and env.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit cannot both be True.")

        token = self.config.api_keys.huggingface_token
        token = token.get_secret_value() if token is not None else None

        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        quantization_config = None

        if env.load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        if env.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=token,
            cache_dir=env.cache_dir,
            trust_remote_code=env.trust_remote_code,
        )

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            token=token,
            cache_dir=env.cache_dir,
            trust_remote_code=env.trust_remote_code,
            device_map=env.device_map,
            dtype=dtype_map[env.dtype],
            quantization_config=quantization_config,
        )

    def _run_huggingface(self, prompt: str, output_model: type[T]) -> T:
        import torch

        if self._hf_model is None or self._hf_tokenizer is None:
            self._load_huggingface_model()

        env = self.config.huggingface_environment
        generation = self.config.generation

        if env is None:
            raise ValueError("huggingface_environment must be configured.")

        messages = [{"role": "user", "content": prompt}]

        inputs = self._hf_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=env.max_model_length is not None,
            max_length=env.max_model_length,
        )

        inputs = {name: tensor.to(self._hf_model.device) for name, tensor in inputs.items()}

        generation_args = {
            "max_new_tokens": generation.max_tokens,
            "do_sample": generation.temperature > 0,
        }

        if generation.temperature > 0:
            generation_args["temperature"] = generation.temperature
            generation_args["top_p"] = generation.top_p

        with torch.no_grad():
            output = self._hf_model.generate(**inputs, **generation_args)

        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
        text = self._hf_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return output_model.model_validate(self._parse_json(text))

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Judge model returned invalid JSON:\n{text}") from error