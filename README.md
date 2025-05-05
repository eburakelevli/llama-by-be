<div align="center">
  <h1>llama-by-be</h1>
  <p><strong>Local chat interface for LLaMA models via Hugging Face Transformers (supports any Hugging Face Transformers model)</strong></p>
  <p>
    <a href="https://github.com/eburakelevli/llama-by-be/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    </a>
    <a href="https://www.python.org">
      <img src="https://img.shields.io/badge/python-3.7%2B-blue.svg" alt="Python Version">
    </a>
  </p>
</div>

## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Chat Interface](#cli-chat-interface)
  - [Web Chat Interface](#web-chat-interface)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About The Project

A simple, open-source chat interface for local models via Hugging Face Transformers (demonstrated with LLaMA models, but compatible with any Transformers model). Includes:

- **CLI**: `chat_llama.py`
- **Web UI**: `web_chat_llama.py` (built with Gradio)

## Features

- Converse with LLaMA models entirely offline
- Supports public & private Hugging Face models
- Configurable sampling: `temperature`, `top_p`, `max_new_tokens`
- Both CLI and web (Gradio) interfaces
- Lightweight and easy to extend

## Prerequisites

- Python 3.7+
- PyTorch
- pandas
- PyPDF2
- Hugging Face Transformers, Accelerate, and SentencePiece

```bash
pip install torch transformers accelerate sentencepiece
```

For the web UI:

```bash
pip install gradio pandas PyPDF2
```

## Installation

```bash
# Clone the repository
git clone https://github.com/eburakelevli/llama-by-be.git
cd llama-by-be
```

Authenticate for private models (optional):

```bash
huggingface-cli login
# or
export HUGGINGFACEHUB_API_TOKEN=<YOUR_TOKEN>
```

## Usage

Replace `<MODEL_ID_OR_PATH>` with the desired Hugging Face model ID or local path.

### CLI Chat Interface

```bash
python chat_llama.py \
  --model <MODEL_ID_OR_PATH> \
  [--token <YOUR_TOKEN>] \
  [--device cpu|cuda|mps] \
  [--max_new_tokens 256] \
  [--temperature 0.7] \
  [--top_p 0.9]
```

### Web Chat Interface

```bash
python web_chat_llama.py \
  --model <MODEL_ID_OR_PATH> \
  [--token <YOUR_TOKEN>] \
  [--device cpu|cuda|mps] \
  [--port 7860] \
  [--share]
```

Open your browser at `http://localhost:7860` (or use the public link if `--share` is set).

- Now supports uploading CSV or PDF files: click the “Upload CSV/PDF” button to attach files, and their contents will be included in your chat context.

## Contributing

> Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to your branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure tests and linting pass before submitting.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://github.com/gradio-app/gradio)
