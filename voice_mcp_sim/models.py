"""The assistant's "brain": pick a model provider with MODEL_PROVIDER.

  bedrock (default)  Amazon Bedrock          BEDROCK_MODEL_ID, AWS_REGION
  groq               Groq (free tier)        GROQ_API_KEY, GROQ_MODEL
  ollama             Ollama on your machine  OLLAMA_HOST, OLLAMA_MODEL
"""

import logging
import os

# Reasoning models behind OpenAI-compatible APIs make Strands warn on every turn; harmless.
logging.getLogger("strands.models.openai").setLevel(logging.ERROR)


def provider() -> str:
    return os.getenv("MODEL_PROVIDER", "bedrock").lower()


def build_model():
    name = provider()
    if name == "groq":
        from strands.models.openai import OpenAIModel
        return OpenAIModel(
            client_args={"api_key": os.environ["GROQ_API_KEY"], "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")},
            model_id=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            params={"temperature": 0.2},
            # Streaming gpt-oss on Groq can repeat the final text and lose tool calls; one full
            # response at a time is reliable, and a voice reply is spoken whole anyway.
            stream=os.getenv("GROQ_STREAM", "off").lower() in {"on", "1", "true"},
        )
    if name == "ollama":
        from strands.models.ollama import OllamaModel
        return OllamaModel(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model_id=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=0.2,
            keep_alive="30m",
        )
    if name == "bedrock":
        from strands.models import BedrockModel
        return BedrockModel(
            model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    raise ValueError(f"Unknown MODEL_PROVIDER '{name}'. Use bedrock, groq or ollama.")
