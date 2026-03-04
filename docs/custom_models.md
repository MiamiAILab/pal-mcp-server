# Custom Models & API Setup

This guide covers setting up custom API endpoints and local model servers. The PAL MCP server supports a unified configuration for all providers through model registries.

## Supported Providers

- **Custom API endpoints** - Local models (Ollama, vLLM, LM Studio, text-generation-webui)
- **Self-hosted APIs** - Any OpenAI-compatible endpoint

## When to Use What

**Use Custom URLs for:**
- **Local models** like Ollama (Llama, Mistral, etc.)
- **Self-hosted inference** with vLLM, LM Studio, text-generation-webui
- **Private/enterprise APIs** that use OpenAI-compatible format
- **Cost control** with local hardware

**Use native APIs (Gemini/OpenAI) when you want:**
- Direct access to specific providers without intermediary
- Potentially lower latency and costs
- Access to the latest model features immediately upon release

**Mix & Match:** You can use multiple providers simultaneously! For example:
- Custom URLs for local models (Ollama Llama)
- Native APIs for specific providers (Gemini Pro with extended thinking)

**Note:** When multiple providers offer the same model name, native APIs take priority.

## Model Aliases

PAL ships multiple registries:

- `conf/openai_models.json` – native OpenAI catalogue (override with `OPENAI_MODELS_CONFIG_PATH`)
- `conf/gemini_models.json` – native Google Gemini catalogue (`GEMINI_MODELS_CONFIG_PATH`)
- `conf/xai_models.json` – native X.AI / GROK catalogue (`XAI_MODELS_CONFIG_PATH`)
- `conf/custom_models.json` – local/self-hosted OpenAI-compatible catalogue (`CUSTOM_MODELS_CONFIG_PATH`)

Copy whichever file you need into your project (or point the corresponding `*_MODELS_CONFIG_PATH` env var at your own copy) and edit it to advertise the models you want.

### Custom/Local Models

| Alias | Maps to Local Model | Note |
|-------|-------------------|------|
| `local-llama`, `local` | `llama3.2` | Requires `CUSTOM_API_URL` configured |

Populate [`conf/custom_models.json`](conf/custom_models.json) with your local models.

Native catalogues (`conf/openai_models.json`, `conf/gemini_models.json`, `conf/xai_models.json`) follow the same schema. Updating those files lets you:

- Expose new aliases (e.g., map `enterprise-pro` to `gpt-5.2-pro`)
- Advertise support for JSON mode or vision if the upstream provider adds it
- Adjust token limits when providers increase context windows

### Latest OpenAI releases

OpenAI's November 13, 2025 drop introduced `gpt-5.1-codex` and `gpt-5.1-codex-mini`, while the flagship base model is now `gpt-5.2`. All of these ship in `conf/openai_models.json`:

| Model | Highlights | Notes |
|-------|------------|-------|
| `gpt-5.2` | 400K context, 128K output, multimodal IO, configurable reasoning effort | Streaming enabled; use for balanced agent/coding flows |
| `gpt-5.1-codex` | Responses-only agentic coding version of GPT-5.1 | Streaming disabled; `use_openai_response_api=true`; `allow_code_generation=true` |
| `gpt-5.1-codex-mini` | Cost-efficient Codex variant | Streaming enabled, retains 400K context and code-generation flag |

These entries include pricing-friendly aliases (`gpt5.2`, `codex-5.1`, `codex-mini`) plus updated capability flags (`supports_extended_thinking`, `allow_code_generation`). Copy the manifest if you operate custom deployment names so downstream providers inherit the same metadata.

Because providers load the manifests on import, you can tweak capabilities without touching Python. Restart the server after editing the JSON files so changes are picked up.

To control ordering in auto mode or the `listmodels` summary, adjust the
[`intelligence_score`](model_ranking.md) for each entry (or rely on the automatic
heuristic described there).

**Note:** Models not in the config file will use generic capabilities (32K context window, no extended thinking, etc.) which may not match the model's actual capabilities. For best results, add new models to the config file with their proper specifications.

## Quick Start

### Custom API Setup (Ollama, vLLM, etc.)

For local models like Ollama, vLLM, LM Studio, or any OpenAI-compatible API:

#### 1. Start Your Local Model Server
```bash
# Example: Ollama
ollama serve
ollama pull llama3.2

# Example: vLLM
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf

# Example: LM Studio (enable OpenAI compatibility in settings)
# Server runs on localhost:1234
```

#### 2. Configure Environment Variables
```bash
# Add to your .env file
CUSTOM_API_URL=http://localhost:11434/v1  # Ollama example
CUSTOM_API_KEY=                                      # Empty for Ollama (no auth needed)
CUSTOM_MODEL_NAME=llama3.2                          # Default model to use
```

**Local Model Connection**

The PAL MCP server runs natively, so you can use standard localhost URLs to connect to local models:

```bash
# For Ollama, vLLM, LM Studio, etc. running on your machine
CUSTOM_API_URL=http://localhost:11434/v1  # Ollama default port
```

#### 3. Examples for Different Platforms

**Ollama:**
```bash
CUSTOM_API_URL=http://localhost:11434/v1
CUSTOM_API_KEY=
CUSTOM_MODEL_NAME=llama3.2
```

**vLLM:**
```bash
CUSTOM_API_URL=http://localhost:8000/v1
CUSTOM_API_KEY=
CUSTOM_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
```

**LM Studio:**
```bash
CUSTOM_API_URL=http://localhost:1234/v1
CUSTOM_API_KEY=lm-studio  # Or any value, LM Studio often requires some key
CUSTOM_MODEL_NAME=local-model
```

**text-generation-webui (with OpenAI extension):**
```bash
CUSTOM_API_URL=http://localhost:5001/v1
CUSTOM_API_KEY=
CUSTOM_MODEL_NAME=your-loaded-model
```

## Using Models

**Using model aliases (from the registry files):**
```
# Local models (with custom URL configured):
"Use local-llama to analyze this code"     # → llama3.2 (local)
"Use local to debug this function"         # → llama3.2 (local)
```

**Using full model names:**
```
# Local/custom models:
"Use llama3.2 via pal to review this"
"Use meta-llama/Llama-2-7b-chat-hf via pal to analyze"
```

**For Local models:** Context window and capabilities are defined in `conf/custom_models.json`.

## Model Provider Selection

The system automatically routes models to the appropriate provider:

1. Entries in `conf/custom_models.json` → Always routed through the Custom API (requires `CUSTOM_API_URL`)
2. **Unknown models** → Fallback logic based on model name patterns

**Provider Priority Order:**
1. Native APIs (Google, OpenAI, X.AI, etc.) - if API keys are available
2. Custom endpoints - for models declared in `conf/custom_models.json`

This ensures clean separation between local and native-API models.

## Model Configuration

These JSON files define model aliases and capabilities. You can:

1. **Use the default configuration** - Includes popular models with convenient aliases
2. **Customize the configuration** - Add your own models and aliases
3. **Override the config path** - Set `CUSTOM_MODELS_CONFIG_PATH` environment variable to an absolute path on disk

### Adding Custom Models

Edit `conf/custom_models.json` to add local models. Each entry maps directly onto [`ModelCapabilities`](../providers/shared/model_capabilities.py).

#### Adding a Custom/Local Model

```json
{
  "model_name": "my-local-model",
  "aliases": ["local-model", "custom"],
  "context_window": 128000,
  "supports_extended_thinking": false,
  "supports_json_mode": false,
  "supports_function_calling": false,
  "description": "My custom Ollama/vLLM model"
}
```

**Field explanations:**
- `model_name`: The model identifier (e.g., local name like `llama3.2`)
- `aliases`: Array of short names users can type instead of the full model name
- `context_window`: Total tokens the model can process (input + output combined)
- `supports_extended_thinking`: Whether the model has extended reasoning capabilities
- `supports_json_mode`: Whether the model can guarantee valid JSON output
- `supports_function_calling`: Whether the model supports function/tool calling
- `description`: Human-readable description of the model

**Important:** Keep models in their respective config files so that requests are routed correctly.

## Troubleshooting

- **"Model not found"**: Check exact model name in your provider's config
- **"Model not available"**: Ensure the provider's API key is configured and the model is in the config
