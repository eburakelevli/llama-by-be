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
  - [Optional: SageMaker Inference](#sagemaker-inference)
  - [Optional: S3-based Model Storage](#optional-s3-based-model-storage)
  - [Optional: Google Sign-In Integration](#google-sign-in-integration)
- [SageMaker Model Files](#sagemaker-model-files)
  - [llama_inference.py](#inferencepy)
  - [Test Points](#test-points)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About The Project

A simple, open-source chat interface for local models via Hugging Face Transformers (demonstrated with LLaMA models, but compatible with any Transformers model). Includes:

- **CLI**: `chat_llama.py`
- **Web UI**: `web_chat_llm.py` (built with Gradio)

## Features

- Converse with LLaMA models entirely offline
- Supports public & private Hugging Face models
- Configurable sampling: `temperature`, `top_p`, `max_new_tokens`
- Both CLI and web (Gradio) interfaces
- Lightweight and easy to extend
- NEW: Now supports uploading CSV or PDF files for Web Chat Interface: click the "Upload CSV/PDF" button to attach files, and their contents will be included in your chat context.


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
python web_chat_llm.py \
  --model <MODEL_ID_OR_PATH> \
  [--token <YOUR_TOKEN>] \
  [--device cpu|cuda|mps] \
  [--sagemaker-endpoint-name <ENDPOINT_NAME>] \
  [--port 7860] \
  [--share]
``` 
 
### SageMaker Inference

You can route inference calls to a deployed AWS SageMaker endpoint instead of loading a local model on your machine.

1. Export your AWS credentials and preferred region:

   ```bash
   export AWS_ACCESS_KEY_ID="your_access_key_id"
   export AWS_SECRET_ACCESS_KEY="your_secret_access_key"
   export AWS_DEFAULT_REGION="your_aws_region"
   ```

2. Launch the web chat application with your SageMaker endpoint:

   Option 1 - Using environment variable:
   ```bash
   export SAGEMAKER_ENDPOINT_NAME="your_sagemaker_endpoint_name"
   python web_chat_llm.py --port 7860
   ```

   Option 2 - Using command line flag:
   ```bash
   python web_chat_llm.py \
     --sagemaker-endpoint-name your_sagemaker_endpoint_name \
     --port 7860
   ```

   Note: When using a SageMaker endpoint, the `--model` parameter is optional since the model is already deployed at the endpoint.

### Setting up AWS Credentials

Ensure you have the necessary AWS credentials and permissions:
- IAM user with SageMaker invoke endpoint permissions
- Properly configured AWS credentials
- Correct region where your SageMaker endpoint is deployed

### Using SageMaker Endpoints

The script supports inference through SageMaker endpoints. You can:
- Use existing endpoints
- Specify custom endpoint names
- Configure inference parameters

### Troubleshooting

Common issues and solutions:
- Invalid signature errors: Check AWS credentials and system time
- Endpoint not found: Verify endpoint name and region
- Permission denied: Check IAM policies

### Optional: S3-based Model Storage

You can optionally store and synchronize your model weights using AWS S3. This is useful for large models that you don't want to bundle in your repository or local disk.

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
     python web_chat_llm.py \
       --s3-model-prefix <YOUR_S3_MODEL_PREFIX> \
       [--download-dir <LOCAL_DIR>] \
       [--device cpu|cuda|mps] \
       [--port 7860]
     ```

   - Upload-only (push a local directory or HF hub model to S3 under `<S3_UPLOAD_PREFIX>`):

     ```bash
     python web_chat_llm.py \
       --model <MODEL_ID_OR_LOCAL_PATH> \
       --upload-to-s3 \
       [--s3-upload-prefix <S3_UPLOAD_PREFIX>]
     ```

   - Sync-if-missing (first check S3, download if present; otherwise upload, then load):

     ```bash
     python web_chat_llm.py \
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
python web_chat_llm_google_signin_test.py \
  --model <MODEL_ID_OR_PATH> \
  [--device cpu|cuda|mps] \
  [--port 7860] \
  [--token <YOUR_HF_TOKEN>]
```

Open your browser at `http://localhost:7860` and click the Google Sign-In button to authenticate before chatting.

## SageMaker Model Files

### llama_inference.py
This file contains the core inference code used by AWS SageMaker to serve the LLaMA model. It defines four essential functions that SageMaker requires:

1. `model_fn(model_dir, context=None)`:
   - Loads the model and tokenizer from the model directory
   - Uses float16 for better performance
   - Automatically handles device placement
   - Returns a dictionary containing both model and tokenizer

2. `input_fn(request_body, request_content_type)`:
   - Processes incoming requests
   - Handles JSON input format
   - Validates content type

3. `predict_fn(input_data, model_dict)`:
   - Generates model responses
   - Uses the following generation parameters:
     - max_new_tokens: 256
     - temperature: 0.7
     - top_p: 0.9
     - do_sample: True
   - Returns generated text in a format compatible with the web chat interface

4. `output_fn(prediction, response_content_type)`:
   - Formats the model's output
   - Returns JSON response

This file is used in the SageMaker model package (`model.tar.gz`) and is essential for the endpoint to function correctly.

### Test Points
The test points file contains example prompts and expected responses used to verify the model's behavior. These test cases help ensure:
- Model loading works correctly
- Tokenizer processes input properly
- Generation parameters produce expected results
- Response formatting is correct

To use these files with SageMaker:
1. Place `inference.py` in the `model/code/` directory
2. Create a model package: `tar -czf model.tar.gz -C model .`
3. Upload to S3: `aws s3 cp model.tar.gz s3://your-bucket/model.tar.gz`
4. Create/update SageMaker model using the container: `763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-inference:2.3.0-transformers4.48.0-gpu-py311-cu121-ubuntu22.04-v2.2`

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
