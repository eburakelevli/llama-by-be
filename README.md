# llama-by-be

## Local Chat Interface

This repository provides a simple command-line chat interface to run a Hugging Face causal language model (e.g., LLaMA 4) locally via Transformers.

### Prerequisites

- Python 3.7 or newer
- PyTorch
- Hugging Face Transformers, Accelerate, and SentencePiece

Install dependencies:

```bash
pip install torch transformers accelerate sentencepiece
```

### Usage

1. (Optional) If you need to authenticate to access a private model, you can:
   - Login via the Hugging Face CLI:
     ```bash
     huggingface-cli login
     ```
   - Or export your token for direct use:
     ```bash
     export HUGGINGFACEHUB_API_TOKEN=<YOUR_TOKEN>
     ```

2. Run the chat script, specifying the model ID or local path. If your model is private, either have logged in via `huggingface-cli` or pass the token directly:

   ```bash
   python chat_llama.py --model meta-llama/Llama-4-7b-chat [--token <YOUR_TOKEN>]
   ```

3. Chat at the `User:` prompt. Type `exit` or `quit` to stop.

## Web UI with Gradio

You can also chat in your browser by running a Gradio-based web app.

1. Install Gradio:
   ```bash
   pip install gradio
   ```

2. Run the web chat script:
   ```bash
   python web_chat_llama.py --model <MODEL_ID_OR_LOCAL_PATH> [--token <YOUR_TOKEN>] [--device cpu] [--port 7860] [--share]
   ```

3. Open your browser to http://localhost:7860 (or the public link if you used `--share`).