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
  - [Optional: S3-based Model Storage](#optional-s3-based-model-storage)
  - [Google Sign-In Integration](#google-sign-in-integration)
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
- NEW: Now supports uploading CSV or PDF files for Web Chat Interface: click the “Upload CSV/PDF” button to attach files, and their contents will be included in your chat context.


## Prerequisites

- Python 3.7+
- PyTorch
- pandas
- PyPDF2
- boto3 (for S3 storage functionality)
- Hugging Face Transformers, Accelerate, and SentencePiece

```bash
pip install torch transformers accelerate sentencepiece
```

For the web UI:

```bash
pip install gradio pandas PyPDF2 boto3
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

### Optional: S3-based Model Storage

You can optionally store and synchronize your model weights using AWS S3. This is useful for large models that you don’t want to bundle in your repository or local disk.

1. Configure your AWS environment variables (never commit these to source control):

   ```bash
   export AWS_ACCESS_KEY_ID="your_access_key_id"
   export AWS_SECRET_ACCESS_KEY="your_secret_access_key"
   export AWS_DEFAULT_REGION="your_bucket_region"
   export S3_BUCKET_NAME="your_bucket_name"
   ```

2. Install required libraries:

   ```bash
   pip install boto3 huggingface-hub
   ```

3. Usage modes:

   - Download-only (load from S3 into a local folder, default: `models/<prefix>`):

     ```bash
     python web_chat_llama.py \
       --s3-model-prefix <YOUR_S3_MODEL_PREFIX> \
       [--download-dir <LOCAL_DIR>] \
       [--device cpu|cuda|mps] \
       [--port 7860]
     ```

   - Upload-only (push a local directory or HF hub model to S3 under `<S3_UPLOAD_PREFIX>`):

     ```bash
     python web_chat_llama.py \
       --model <MODEL_ID_OR_LOCAL_PATH> \
       --upload-to-s3 \
       [--s3-upload-prefix <S3_UPLOAD_PREFIX>]
     ```

   - Sync-if-missing (first check S3, download if present; otherwise upload, then load):

     ```bash
     python web_chat_llama.py \
       --model <MODEL_ID_OR_LOCAL_PATH> \
       --s3-model-prefix <YOUR_S3_MODEL_PREFIX> \
       --upload-to-s3 \
       [--download-dir <LOCAL_DIR>] \
       [--device cpu|cuda|mps] \
       [--port 7860]
     ```

4. Default behavior (no S3 flags) is to load directly from `--model` (Hugging Face hub ID or local path).

Open your browser at `http://localhost:7860` (or use the public link if `--share` is set).

### Google Sign-In Integration

Follow these steps to enable Google Sign-In in the Web Chat interface:

1. Create OAuth 2.0 Client ID:
    - Go to the Google Cloud Console: https://console.cloud.google.com/apis/credentials
    - Select your project or create a new one.
    - Enable the "Google Identity Services" API.
    - Click "Create Credentials" > "OAuth client ID".
    - Choose "Web application" and set the following:
        - Authorized JavaScript origins: http://localhost:7860
        - Authorized redirect URIs: http://localhost:7860/api/auth/google
    - Click "Create" and copy the **Client ID** and **Client Secret**.

2. Set environment variables:
```bash
export GOOGLE_CLIENT_ID="YOUR_CLIENT_ID"
export GOOGLE_CLIENT_SECRET="YOUR_CLIENT_SECRET"
export OAUTH_REDIRECT_URI="http://localhost:7860/api/auth/google"
```

3. Install required dependencies (if not already installed):
```bash
pip install google-auth fastapi uvicorn
```

4. Run the Web Chat with Google Sign-In support:
```bash
python web_chat_llama_google_signin_test.py \
  --model <MODEL_ID_OR_PATH> \
  [--device cpu|cuda|mps] \
  [--port 7860] \
  [--token <YOUR_HF_TOKEN>]
```

Open your browser at `http://localhost:7860` and click the Google Sign-In button to authenticate before chatting.


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
