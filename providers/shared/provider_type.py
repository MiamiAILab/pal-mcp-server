"""Enumeration describing which backend owns a given model."""

from enum import Enum

__all__ = ["ProviderType"]


class ProviderType(Enum):
    """Canonical identifiers for every supported provider backend."""

    GOOGLE = "google"
    OPENAI = "openai"
    XAI = "xai"
    CUSTOM = "custom"
    MINIMAX = "minimax"
    MOONSHOT = "moonshot"
    ZHIPU = "zhipu"
    ALIBABA = "alibaba"
    # TOGETHER purged 2026-07-04 (GENESIS-097) — Together retired serverless Qwen
    PERPLEXITY = "perplexity"
    MISTRAL = "mistral"
    OPENROUTER = "openrouter"
