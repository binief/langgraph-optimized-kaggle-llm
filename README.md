# LangGraph-Optimized Kaggle LLM Gateway

Run **`Qwen3.8-27B`** on Kaggle (`GPU T4 x2`) using prebuilt **`llama.cpp`** (`llama-server`) paired with a **LangGraph + FastAPI** orchestration gateway for VS Code coding agents (Cline, Roo Code, Continue, Kilo Code, OpenCode) and OpenAI-compatible clients.

## Notebooks
- **`qwen38_27b_vscode_gateway.ipynb`** *(Recommended / latest)* — Dual-model VS Code coding gateway:
  - **Solver (`:8000` on GPU):** `Qwen3.8-27B-UD-IQ4_XS` (`32K` context, `-np 1`, quantized KV cache, Flash Attention, Jinja).
  - **Compactor / Router (`:8001` on GPU 1 or CPU):** `Qwen3-1.7B-Q8_0` (`8K` context).
  - **LangGraph Gateway (`:8080`):** tool-output pruning, async context compaction with SQLite summary caching, a file/error ledger, Qwen-compatible system-message normalization, loop detection, lightweight request offloading, SSE keepalives, and `/metrics`.
- **`qwen38_27b_langgraph_demo.ipynb`** — Multi-step LangGraph tool-calling demo with mock REST APIs.
- **`qwen38_27b_openai_server.ipynb`** — Minimal prebuilt `llama.cpp` server for `Qwen3.8-27B` with Cloudflare Tunnel.
- **`ligguf_openai_server.ipynb`** — Lightweight CPU-only LiGGUF server for smaller LLaMA/Qwen3 GGUF models.

## Python scripts
Application code is kept in the repository's `scripts/` directory rather than embedded in notebook cells:

- `scripts/langgraph_vscode_gateway.py` — recommended VS Code-agent gateway.
- `scripts/langgraph_gateway.py` and `scripts/mock_api.py` — multi-step LangGraph demo.
- `scripts/ligguf_server.py` — LiGGUF OpenAI-compatible API server.

The notebooks that use these scripts clone this repository into `/kaggle/working/langgraph-optimized-kaggle-llm` and launch the scripts from that checkout. Keep **Internet: ON** in Kaggle so cloning the repository and downloading the model/dependencies work.

## Quick Start (Kaggle)
1. Import `qwen38_27b_vscode_gateway.ipynb` into Kaggle.
2. Enable **Accelerator: GPU T4 x2** and **Internet: ON**.
3. Run all cells top to bottom.
4. Copy the `https://*.trycloudflare.com` URL printed in Section 7 and set your VS Code agent to:
   - **Provider:** OpenAI Compatible
   - **Base URL:** `https://<your-tunnel>.trycloudflare.com/v1`
   - **Model ID:** `qwen3.8-27b`
   - **API Key:** `not-needed`
