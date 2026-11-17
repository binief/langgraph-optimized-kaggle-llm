# LangGraph-Optimized Kaggle LLM Gateway

Run **`Qwen3.8-27B`** on Kaggle (`GPU T4 x2`) using **prebuilt `llama.cpp`** (`llama-server`) paired with a **LangGraph + FastAPI** orchestration gateway designed for **VS Code coding agents** (Cline, Roo Code, Continue, Kilo Code, OpenCode) and OpenAI-compatible clients.

## Notebooks Included
- **`qwen38_27b_vscode_langgraph_kaggle.ipynb`** *(Recommended)* — Dual-model VS Code coding gateway:
  - **Solver (`:8000` on GPU):** `Qwen3.8-27B-UD-IQ4_XS` (`32K` ctx, `-np 1`, `--cache-type-k/v q8_0`, `-fa on`, `--jinja`).
  - **Compactor / Router (`:8001` on GPU 1 / CPU):** `Qwen3-1.7B-Q8_0` (`8K` ctx).
  - **LangGraph Gateway (`:8080`):**
    - Smart tool-output pruning (keeps head/tail + error lines on older tool outputs).
    - Background async pre-compaction (`60%` soft watermark -> `75%` hard watermark -> `40%` tail) with SQLite summary caching.
    - Deterministic File & Error Ledger pinned alongside summaries.
    - Stuck tool-loop breaker & error-aware dynamic sampling (`temperature` + `enable_thinking`).
    - Real-time intra-stream **thinking-loop detector** with automatic `enable_thinking=False` recovery.
    - Lightweight request offloading (title/summary requests -> `:8001`), SSE keepalives, and `/metrics`.
- **`qwen38_27b_langgraph_kaggle.ipynb`** — Multi-step LangGraph tool-calling demo with mock REST APIs.
- **`qwen38_27b_kaggle_openai.ipynb`** — Minimal prebuilt `llama.cpp` server for `Qwen3.8-27B` with Cloudflare Tunnel.
- **`ligguf_kaggle_openai.ipynb`** — Lightweight CPU-only `LiGGUF` server for smaller LLaMA/Qwen3 GGUF models.

## Quick Start (Kaggle)
1. Import `qwen38_27b_vscode_langgraph_kaggle.ipynb` into Kaggle.
2. Enable **Accelerator: GPU T4 x2** and **Internet: ON**.
3. Run all cells top to bottom.
4. Copy the `https://*.trycloudflare.com` URL printed in Section 7 and set your VS Code agent to:
   - **Provider:** OpenAI Compatible
   - **Base URL:** `https://<your-tunnel>.trycloudflare.com/v1`
   - **Model ID:** `qwen3.8-27b`
   - **API Key:** `not-needed`
