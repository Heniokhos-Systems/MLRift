# Model weights

This directory is the default search location for the model files that the
drivers in `examples/` need. Files placed here are **gitignored** — drop
GGUF / safetensors blobs in and they will not be tracked.

Each driver resolves its model path via `std/model_path.mlr` →
`resolve_model_path(env_var, default_path, driver_name)`:

1. If `$env_var` is set and points to a readable file, that path is used.
2. Otherwise, `default_path` (the entry under `Default file` below) is
   tried, resolved relative to the process working directory. So either
   run from the repo root or set the env var to an absolute path.
3. If neither works, the driver prints a clear error and exits.

## Expected files per driver

| Driver                                | Env var                       | Default file (under `models/`)                         | Source |
|---------------------------------------|-------------------------------|--------------------------------------------------------|--------|
| `examples/llama3_generate.mlr`        | `MLRIFT_LLAMA3_1B_GGUF`       | `models/llama-3.2-1b-instruct-q8_0.gguf`               | `ollama pull llama3.2:1b` (Q8_0); blob lives under `~/.ollama/models/blobs/sha256-*` — symlink into `models/`. |
| `examples/llama3_1b_gpu_generate.mlr` | `MLRIFT_LLAMA3_1B_GGUF`       | `models/llama-3.2-1b-instruct-q8_0.gguf`               | same blob as the CPU driver. |
| `examples/llama3_3b_generate.mlr`     | `MLRIFT_LLAMA3_3B_GGUF`       | `models/llama-3.2-3b-instruct-q8_0.gguf`               | `ollama pull llama3.2:3b` (Q8_0). |
| `examples/mistral_generate.mlr`       | `MLRIFT_MISTRAL_7B_GGUF`      | `models/mistral-7b-instruct-v0.3-q8_0.gguf`            | `ollama pull mistral:7b-instruct-v0.3-q8_0`. |
| `examples/gemma2_2b_generate.mlr`     | `MLRIFT_GEMMA2_2B_GGUF`       | `models/gemma-2-2b-it-q8_0.gguf`                       | `ollama pull gemma2:2b` (Q8_0). |
| `examples/gemma3_1b_generate.mlr`     | `MLRIFT_GEMMA3_1B_GGUF`       | `models/gemma-3-1b-it-q8_0.gguf`                       | `ollama pull gemma3:1b` (Q8_0). |
| `examples/qwen36_4b_generate.mlr`     | `MLRIFT_QWEN3VL_4B_GGUF`      | `models/qwen3-vl-4b-instruct-q8_0.gguf`                | Hugging Face `Qwen/Qwen3-VL-4B-Instruct-GGUF`. |
| `examples/qwen35_0_8b_generate.mlr`   | `MLRIFT_QWEN35_0_8B_GGUF`     | `models/qwen3.5-0.8b-q8_0.gguf`                        | Hugging Face `Qwen/Qwen3.5-0.8B-GGUF`. |
| `examples/qwen3_14b_q8_generate.mlr`  | `MLRIFT_QWEN3_14B_GGUF`       | `models/qwen3-14b-q8_0.gguf`                           | Hugging Face `Qwen/Qwen3-14B-GGUF`. |
| `examples/qwen3_14b_q8_generate.mlr`  | `MLRIFT_QWEN3_TOKENIZER_JSON` | `models/qwen3-tokenizer.json`                          | shared Qwen3 tokenizer JSON. |
| `examples/qwen3_generate*.mlr`        | `MLRIFT_QWEN3_0_6B_DIR`       | `models/qwen3-0.6b/` (contains `model.safetensors` etc.) | Hugging Face `Qwen/Qwen3-0.6B`. |
| `examples/qwen3_generate_gguf.mlr`    | `MLRIFT_QWEN3_0_6B_GGUF`      | `models/qwen3-0.6b-bf16.gguf`                          | Hugging Face `Qwen/Qwen3-0.6B-GGUF`. |

Probes (`gguf_probe.mlr`, `qwen35_probe.mlr`, etc.) reuse the same env
vars as their corresponding driver.

## Tip — symlinking from `ollama`

If you already have a model cached via `ollama pull`, symlink the blob
into `models/` rather than copying:

```bash
# Find the blob hash for, e.g., llama3.2:1b
ollama show --modelfile llama3.2:1b | grep -E '^FROM'

ln -s /usr/share/ollama/.ollama/models/blobs/sha256-<hash> \
      models/llama-3.2-1b-instruct-q8_0.gguf
```
