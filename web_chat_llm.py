#!/usr/bin/env python3
"""
Web-based chat interface to interact with a local Hugging Face causal language model using Gradio.
Supports conversational chat with any compatible model (e.g., LLaMA 4).
"""
import sys
import argparse
import os
import json
import urllib.parse
import boto3
import botocore
import datetime
import hashlib
import hmac

# Essential imports needed for both local and SageMaker modes
try:
    import gradio as gr
    import pandas as pd
    from PyPDF2 import PdfReader
except ImportError as e:
    print(f"Error: Required library not found: {e}", file=sys.stderr)
    sys.exit(1)

# Only import torch if not using SageMaker endpoint
torch = None
transformers = None

def load_model(model_name, device, hf_token=None):
    if not torch or not transformers:
        raise RuntimeError("PyTorch and Transformers are required for local model inference")
    params = dict(
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    if hf_token:
        params['token'] = hf_token
    print(f"Loading model '{model_name}' on {device}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        **({'token': hf_token} if hf_token else {})
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **params)
    if device in ("cpu", "mps"):
        model.to(device)
    model.eval()
    return tokenizer, model

def build_prompt(history, file_store=None):
    prompt = ""
    # include any uploaded file contents first
    if file_store:
        for name, text in file_store:
            prompt += f"[Content from {name}]\n{text}\n\n"
    # build convo from history of dict messages
    for msg in history:
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'user':
            prompt += f"User: {content}\n"
        elif role == 'assistant':
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant:"
    return prompt

def respond(message, history, tokenizer, model, device, max_new_tokens, temperature, top_p, file_store=None):
    if not torch:
        raise RuntimeError("PyTorch is required for local model inference")
    history = history or []
    prompt_history = history + [{"role": "user", "content": message}]
    prompt = build_prompt(prompt_history, file_store)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    if "\nUser:" in text:
        text = text.split("\nUser:")[0].strip()
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": text}
    ]
    return new_history, ""

def setup_aws_credentials():
    """Set up AWS credentials either from environment variables or user input."""
    # Check existing credentials
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')

    if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
        print("\nAWS credentials not found in environment. Please enter them now:")
        if not aws_access_key_id:
            aws_access_key_id = input("AWS Access Key ID: ").strip()
            os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        if not aws_secret_access_key:
            aws_secret_access_key = input("AWS Secret Access Key: ").strip()
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        if not aws_region:
            aws_region = input("AWS Region (default: eu-west-1): ").strip() or "eu-west-1"
            os.environ['AWS_DEFAULT_REGION'] = aws_region
    
    # Print debug info for credentials
    print(f"\nUsing AWS credentials:")
    print(f"  Access Key ID: {aws_access_key_id[:4]}... (masked for security)")
    print(f"  Secret Access Key: {aws_secret_access_key[:4]}... (masked for security)")
    print(f"  Region: {aws_region}")

    return aws_access_key_id, aws_secret_access_key, aws_region

def invoke_sagemaker_endpoint(endpoint_name, payload, region=None, access_key=None, secret_key=None):
    """
    Invoke a SageMaker endpoint using the AWS CLI directly.
    This bypasses all signature issues by delegating to the AWS CLI.
    """
    import subprocess
    import tempfile
    import json
    
    # Set environment variables for AWS CLI
    env = os.environ.copy()
    env['AWS_ACCESS_KEY_ID'] = access_key
    env['AWS_SECRET_ACCESS_KEY'] = secret_key
    env['AWS_DEFAULT_REGION'] = region

    # If AWS_SESSION_TOKEN is in the parent environment, make sure it's passed
    if 'AWS_SESSION_TOKEN' in os.environ:
        env['AWS_SESSION_TOKEN'] = os.environ['AWS_SESSION_TOKEN']
    elif 'AWS_SESSION_TOKEN' in env:
        del env['AWS_SESSION_TOKEN']

    # Prevent AWS CLI from using profiles or other local AWS config files
    env['AWS_CONFIG_FILE'] = '/dev/null'
    env['AWS_SHARED_CREDENTIALS_FILE'] = '/dev/null'
    
    if 'AWS_PROFILE' in env:
        del env['AWS_PROFILE']
    
    # Create a temporary file for the payload
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp:
        temp_path = temp.name
        json.dump(payload, temp)
    
    output_path = ""
    try:
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as outfile:
            output_path = outfile.name
        
        # Build the AWS CLI command
        cmd = [
            "aws", "sagemaker-runtime", "invoke-endpoint",
            "--endpoint-name", endpoint_name,
            "--content-type", "application/json",
            "--body", f"fileb://{temp_path}",
            "--cli-read-timeout", "300",
            "--cli-connect-timeout", "60",
            output_path 
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        print(f"CLI Environment AWS_ACCESS_KEY_ID: {env.get('AWS_ACCESS_KEY_ID', 'Not Set')[:4]}...")
        print(f"CLI Environment AWS_SECRET_ACCESS_KEY: {env.get('AWS_SECRET_ACCESS_KEY', 'Not Set')[:4]}...")
        print(f"CLI Environment AWS_SESSION_TOKEN: {'Present' if env.get('AWS_SESSION_TOKEN') else 'Not Present'}")
        print(f"CLI Environment AWS_DEFAULT_REGION: {env.get('AWS_DEFAULT_REGION', 'Not Set')}")

        # Call AWS CLI and capture output
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=360
        )
        
        # Read the output file
        with open(output_path, 'r') as f:
            response_body = f.read()
        
        # Parse the response
        try:
            response_data = json.loads(response_body)
            return response_data
        except json.JSONDecodeError:
            return {"response": response_body}
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"AWS CLI timeout after 360 seconds: {e}. STDERR: {e.stderr if hasattr(e, 'stderr') else 'No stderr'} STDOUT: {e.stdout if hasattr(e, 'stdout') else 'No stdout'}"
        print(f"Error: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"AWS CLI error: {e}. STDERR: {e.stderr} STDOUT: {e.stdout}"
        print(f"Error: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)

def main():
    parser = argparse.ArgumentParser(description="Web chat with a local Hugging Face model via Gradio")
    parser.add_argument("-m", "--model",
                        help="Model ID or local path (e.g., meta-llama/Llama-4-7b-chat). Required unless using SageMaker endpoint.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"],
                        help="Device to run on (default: mps if available, else cuda if available, else cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling probability")
    parser.add_argument("--token", help="Hugging Face token for private models (overrides CLI login)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for the Gradio server")
    parser.add_argument("--share", action="store_true",
                        help="Create a public link for the Gradio app")
    parser.add_argument("--s3-model-prefix",
                        help="S3 prefix or key for the model (object key for archive or prefix for model directory). Requires S3_BUCKET_NAME env var")
    parser.add_argument("--download-dir",
                        help="Local directory to download model from S3 (default: models/<prefix>)")
    parser.add_argument("--upload-to-s3", action="store_true",
                        help="Upload model from Hugging Face or local path to S3 bucket")
    parser.add_argument("--s3-upload-prefix",
                        help="S3 key prefix under which to upload the model (defaults to model name)")
    parser.add_argument("--sagemaker-endpoint-name", dest="sagemaker_endpoint_name",
                        help="SageMaker endpoint name for inference. If set, skips local model and calls SageMaker endpoint.")
    parser.add_argument("--cli-read-timeout", type=int, default=300,
                        help="Timeout in seconds for SageMaker endpoint read operations (default: 300)")
    parser.add_argument("--cli-connect-timeout", type=int, default=60,
                        help="Timeout in seconds for SageMaker endpoint connection (default: 60)")
    parser.add_argument("--subprocess-timeout", type=int, default=360,
                        help="Timeout in seconds for the entire subprocess (default: 360)")
    args = parser.parse_args()
    
    # Check for optional SageMaker endpoint (use SAGEMAKER_ENDPOINT_NAME env var)
    endpoint_name = args.sagemaker_endpoint_name or os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    
    # Validate arguments
    if not endpoint_name and not args.model:
        parser.error("--model is required when not using a SageMaker endpoint")

    if endpoint_name:
        # Verify AWS credentials are present and not empty
        required_vars = {
            'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID', '').strip(),
            'AWS_SECRET_ACCESS_KEY': os.environ.get('AWS_SECRET_ACCESS_KEY', '').strip(),
            'AWS_DEFAULT_REGION': os.environ.get('AWS_DEFAULT_REGION', '').strip(),
            'SAGEMAKER_ENDPOINT_NAME': os.environ.get('SAGEMAKER_ENDPOINT_NAME', '').strip()
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            print(f"Error: Missing or empty environment variables: {', '.join(missing_vars)}", file=sys.stderr)
            sys.exit(1)
            
        print(f"\nUsing SageMaker endpoint: {required_vars['SAGEMAKER_ENDPOINT_NAME']}")
        print(f"AWS Region: {required_vars['AWS_DEFAULT_REGION']}")
        use_sagemaker = True
    else:
        use_sagemaker = False

    # Handle S3 <-> local model sync/upload/download
    if args.upload_to_s3 or args.s3_model_prefix:
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            print("Error: S3 operation requires S3_BUCKET_NAME env var", file=sys.stderr)
            sys.exit(1)
        try:
            import boto3
        except ImportError:
            print("Error: boto3 is required for S3 operations. Install with 'pip install boto3'", file=sys.stderr)
            sys.exit(1)
        s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        )
        # Sync logic: if both upload and download flags are set
        if args.upload_to_s3 and args.s3_model_prefix:
            prefix = args.s3_model_prefix
            # Check if model exists in S3
            exists = False
            res = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
            if res.get('KeyCount', 0) > 0 or res.get('Contents'):
                exists = True
            if exists:
                # download existing model
                print(f"Model found in S3 at prefix '{prefix}', downloading...", file=sys.stderr)
                # reuse download logic
                # Determine local dir
                if args.download_dir:
                    local_model_dir = args.download_dir
                else:
                    base = os.path.basename(prefix.rstrip("/"))
                    local_model_dir = os.path.join("models", base)
                if os.path.exists(local_model_dir):
                    print(f"Local model directory {local_model_dir} already exists, skipping download.", file=sys.stderr)
                else:
                    # download objects
                    os.makedirs(local_model_dir, exist_ok=True)
                    paginator = s3.get_paginator("list_objects_v2")
                    for result in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                        for obj in result.get("Contents", []):
                            key = obj["Key"]
                            rel = os.path.relpath(key, prefix)
                            dest = os.path.join(local_model_dir, rel)
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            print(f"Downloading s3://{bucket_name}/{key} to {dest}", file=sys.stderr)
                            s3.download_file(bucket_name, key, dest)
                args.model = local_model_dir
            else:
                # upload local or HF model to S3
                print(f"Model not found in S3 at prefix '{prefix}', uploading...", file=sys.stderr)
                # get local model dir
                if os.path.isdir(args.model):
                    local_model_dir = args.model
                else:
                    try:
                        from huggingface_hub import snapshot_download
                    except ImportError:
                        print("Error: huggingface_hub is required to download model. Install with 'pip install huggingface-hub'", file=sys.stderr)
                        sys.exit(1)
                    local_model_dir = snapshot_download(repo_id=args.model, token=args.token)
                upload_prefix = args.s3_upload_prefix or prefix
                for root, _, files in os.walk(local_model_dir):
                    for fname in files:
                        full = os.path.join(root, fname)
                        rel = os.path.relpath(full, local_model_dir)
                        key = f"{upload_prefix}/{rel}".replace(os.sep, "/")
                        print(f"Uploading {full} -> s3://{bucket_name}/{key}", file=sys.stderr)
                        s3.upload_file(full, bucket_name, key)
                args.model = local_model_dir
        elif args.s3_model_prefix:
            # download-only
            prefix = args.s3_model_prefix
            if args.download_dir:
                local_model_dir = args.download_dir
            else:
                base = os.path.basename(prefix.rstrip("/"))
                local_model_dir = os.path.join("models", base)
            if os.path.exists(local_model_dir):
                print(f"Local model directory {local_model_dir} already exists, skipping download.", file=sys.stderr)
            else:
                os.makedirs(local_model_dir, exist_ok=True)
                paginator = s3.get_paginator("list_objects_v2")
                for result in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                    for obj in result.get("Contents", []):
                        key = obj["Key"]
                        rel = os.path.relpath(key, prefix)
                        dest = os.path.join(local_model_dir, rel)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        print(f"Downloading s3://{bucket_name}/{key} to {dest}", file=sys.stderr)
                        s3.download_file(bucket_name, key, dest)
            args.model = local_model_dir
        elif args.upload_to_s3:
            # upload-only
            # get local model dir
            if os.path.isdir(args.model):
                local_model_dir = args.model
            else:
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    print("Error: huggingface_hub is required to download model. Install with 'pip install huggingface-hub'", file=sys.stderr)
                    sys.exit(1)
                local_model_dir = snapshot_download(repo_id=args.model, token=args.token)
            upload_prefix = args.s3_upload_prefix or os.path.basename(local_model_dir.rstrip("/"))
            for root, _, files in os.walk(local_model_dir):
                for fname in files:
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, local_model_dir)
                    key = f"{upload_prefix}/{rel}".replace(os.sep, "/")
                    print(f"Uploading {full} -> s3://{bucket_name}/{key}", file=sys.stderr)
                    s3.upload_file(full, bucket_name, key)
            print("Upload complete.", file=sys.stderr)
            sys.exit(0)


    # Initialize local model unless using SageMaker endpoint
    if not use_sagemaker:
        if not args.model:
            print("Error: --model is required when not using SageMaker endpoint", file=sys.stderr)
            sys.exit(1)
        # Determine device: prefer user-specified, else mps > cuda > cpu
        if args.device:
            device = args.device
        else:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        hf_token = args.token if args.token else None
        tokenizer, model = load_model(args.model, device, hf_token)
    else:
        # Will use SageMaker endpoint for inference
        tokenizer = None
        model = None
        device = None

    # launch Gradio interface
    with gr.Blocks() as demo:
        # display the loaded model name as the chat title
        # use 'messages' format (openai-style dicts) for chat history
        chatbot = gr.Chatbot(label=args.model, type='messages')
        # state to store uploaded file contents
        file_store = gr.State([])
        # file upload component for CSV and PDF
        file_input = gr.File(label="Upload CSV/PDF", file_types=[".csv", ".pdf"], file_count="multiple")

        def process_file(files, history, store):
            history = history or []
            store = store or []
            for f in files:
                file_path = f.name if hasattr(f, 'name') else f
                ext = os.path.splitext(file_path)[1].lower()
                text = ""
                if ext == ".csv":
                    try:
                        df = pd.read_csv(file_path)
                        text = df.to_csv(index=False)
                    except Exception as e:
                        # record error as assistant message
                        history.append({"role": "assistant", "content": f"Error reading CSV {file_path}: {e}"})
                        continue
                elif ext == ".pdf":
                    try:
                        reader = PdfReader(file_path)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                    except Exception as e:
                        history.append({"role": "assistant", "content": f"Error reading PDF {file_path}: {e}"})
                        continue
                else:
                    continue
                basename = os.path.basename(file_path)
                store.append((basename, text))
                # acknowledge upload as assistant message
                history.append({"role": "assistant", "content": f"Uploaded {basename}"})
            # clear file_input by returning None for that component
            return history, store, None

        # handle file uploads
        file_input.upload(process_file, [file_input, chatbot, file_store], [chatbot, file_store, file_input])

        user_input = gr.Textbox(show_label=False, placeholder="Type your message and press enter")
        # define callback including file context (choose SageMaker or local inference)
        if use_sagemaker:
            def gr_respond(message, history, store):
                history = history or []
                store = store or []
                prompt_history = history + [{"role": "user", "content": message}]
                prompt = build_prompt(prompt_history, store)
                payload = {"inputs": prompt}
                
                try:
                    response = invoke_sagemaker_endpoint(
                        os.environ['SAGEMAKER_ENDPOINT_NAME'],
                        payload,
                        os.environ['AWS_DEFAULT_REGION'],
                        os.environ['AWS_ACCESS_KEY_ID'],
                        os.environ['AWS_SECRET_ACCESS_KEY']
                    )
                    # Handle different response formats
                    if isinstance(response, list) and response:
                        text = response[0].get("generated_text", "")
                    elif isinstance(response, dict):
                        # Try different common response formats
                        text = (
                            response.get("generated_text") or
                            response.get("response") or
                            response.get("text") or
                            response.get("output") or
                            ""
                        )
                    else:
                        text = str(response)
                        
                except Exception as e:
                    error_msg = f"Error calling SageMaker endpoint: {str(e)}"
                    print(error_msg, file=sys.stderr)
                    new_history = history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": error_msg}
                    ]
                    return new_history, ""
                
                if "\nUser:" in text:
                    text = text.split("\nUser:")[0].strip()
                
                new_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": text}
                ]
                return new_history, ""
        else:
            def gr_respond(message, history, store):
                return respond(
                    message, history, tokenizer, model, device,
                    args.max_new_tokens, args.temperature, args.top_p, store
                )
        # submit text inputs with access to file_store
        user_input.submit(gr_respond, [user_input, chatbot, file_store], [chatbot, user_input])
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()