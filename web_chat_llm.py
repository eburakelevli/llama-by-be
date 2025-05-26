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
    """Invoke SageMaker endpoint using boto3."""
    try:
        # Print environment information for debugging
        print("\nEnvironment Information:", file=sys.stderr)
        print(f"Python version: {sys.version}", file=sys.stderr)
        print(f"boto3 version: {boto3.__version__}", file=sys.stderr)
        print(f"botocore version: {botocore.__version__}", file=sys.stderr)
        print(f"Running on Heroku: {'DYNO' in os.environ}", file=sys.stderr)
        
        # Get credentials from environment variables only (more reliable on Heroku)
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID', '').strip()
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', '').strip()
        aws_region = os.environ.get('AWS_DEFAULT_REGION', '').strip()

        if not all([aws_access_key, aws_secret_key, aws_region]):
            raise ValueError("Missing required AWS credentials in environment variables")

        # Print credential information (safely)
        print("\nCredential Information:", file=sys.stderr)
        print(f"AWS Access Key ID (first 4 chars): {aws_access_key[:4] if aws_access_key else 'None'}", file=sys.stderr)
        print(f"AWS Secret Key length: {len(aws_secret_key) if aws_secret_key else 0}", file=sys.stderr)
        print(f"AWS Region: {aws_region}", file=sys.stderr)

        # Format the payload according to the endpoint's expectations
        formatted_payload = {
            "inputs": payload["inputs"],
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        # Convert payload to JSON string and calculate SHA256
        payload_json = json.dumps(formatted_payload)
        payload_bytes = payload_json.encode('utf-8')
        content_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        
        # Create timestamp for request
        amz_date = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        date_stamp = amz_date[:8]
        
        # Create canonical request
        canonical_uri = f'/endpoints/{endpoint_name}/invocations'
        canonical_querystring = ''
        
        # Headers must be in alphabetical order
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'host': f'runtime.sagemaker.{aws_region}.amazonaws.com',
            'x-amz-content-sha256': content_sha256,
            'x-amz-date': amz_date
        }
        
        # Create canonical headers (must be in alphabetical order)
        canonical_headers = '\n'.join(f'{k}:{v}' for k, v in sorted(headers.items())) + '\n'
        
        # Create signed headers (must be in alphabetical order)
        signed_headers = ';'.join(sorted(headers.keys()))
        
        # Create canonical request
        canonical_request = (
            'POST\n'
            f'{canonical_uri}\n'
            f'{canonical_querystring}\n'
            f'{canonical_headers}\n'
            f'{signed_headers}\n'
            f'{content_sha256}'
        )
        
        # Create string to sign
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f'{date_stamp}/{aws_region}/sagemaker/aws4_request'
        string_to_sign = (
            f'{algorithm}\n'
            f'{amz_date}\n'
            f'{credential_scope}\n'
            f'{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'
        )
        
        # Calculate signing key
        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
        
        k_date = sign(('AWS4' + aws_secret_key).encode('utf-8'), date_stamp)
        k_region = sign(k_date, aws_region)
        k_service = sign(k_region, 'sagemaker')
        k_signing = sign(k_service, 'aws4_request')
        
        # Calculate signature
        signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Create authorization header
        authorization_header = (
            f'{algorithm} '
            f'Credential={aws_access_key}/{credential_scope}, '
            f'SignedHeaders={signed_headers}, '
            f'Signature={signature}'
        )
        
        # Add authorization header to headers
        headers['authorization'] = authorization_header
        
        # Print request details for debugging (without sensitive info)
        print(f"\nMaking request to endpoint: {endpoint_name}", file=sys.stderr)
        print(f"Region: {aws_region}", file=sys.stderr)
        print(f"Endpoint URL: https://runtime.sagemaker.{aws_region}.amazonaws.com{canonical_uri}", file=sys.stderr)
        print(f"Canonical Request:\n{canonical_request}", file=sys.stderr)
        print(f"String to Sign:\n{string_to_sign}", file=sys.stderr)
        print(f"Request headers (without Authorization): {dict((k,v) for k,v in headers.items() if k != 'authorization')}", file=sys.stderr)
        print(f"Request payload: {payload_json}", file=sys.stderr)
        print(f"Content SHA256: {content_sha256}", file=sys.stderr)
        print(f"Date Stamp: {date_stamp}", file=sys.stderr)
        print(f"AMZ Date: {amz_date}", file=sys.stderr)
        print(f"Credential Scope: {credential_scope}", file=sys.stderr)
        print(f"Signed Headers: {signed_headers}", file=sys.stderr)

        # Make the request using requests library
        import requests
        response = requests.post(
            f'https://runtime.sagemaker.{aws_region}.amazonaws.com{canonical_uri}',
            headers=headers,
            data=payload_bytes
        )
        
        if response.status_code != 200:
            error_body = response.text
            print(f"Error response: {error_body}", file=sys.stderr)
            if 'Canonical String' in error_body and 'String-to-Sign' in error_body:
                print("\nAWS Expected Values:", file=sys.stderr)
                print(f"AWS Expected Canonical String:\n{error_body.split('Canonical String for this request should have been')[1].split('String-to-Sign')[0].strip()}", file=sys.stderr)
                print(f"AWS Expected String-to-Sign:\n{error_body.split('String-to-Sign should have been')[1].split('}')[0].strip()}", file=sys.stderr)
                print("\nOur Calculated Values:", file=sys.stderr)
                print(f"Our Canonical String:\n{canonical_request}", file=sys.stderr)
                print(f"Our String-to-Sign:\n{string_to_sign}", file=sys.stderr)
            raise botocore.exceptions.ClientError(
                error_response={'Error': {'Code': 'InvokeEndpointError', 'Message': error_body}},
                operation_name='InvokeEndpoint'
            )

        # Parse response
        response_body = response.text
        print(f"Raw response: {response_body}", file=sys.stderr)
        return json.loads(response_body)

    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"\nAWS Error ({error_code}): {error_message}", file=sys.stderr)
        if error_code == 'InvalidSignatureException':
            print("\nDetailed credential information:", file=sys.stderr)
            print(f"Access Key ID: {aws_access_key[:4]}...", file=sys.stderr)
            print(f"Secret Key length: {len(aws_secret_key)}", file=sys.stderr)
            print(f"Region: {aws_region}", file=sys.stderr)
            print(f"Canonical Request:\n{canonical_request}", file=sys.stderr)
            print(f"String to Sign:\n{string_to_sign}", file=sys.stderr)
            print("\nPlease verify:", file=sys.stderr)
            print("1. AWS credentials are correctly set in Heroku config vars", file=sys.stderr)
            print("2. The credentials have permissions to invoke the SageMaker endpoint", file=sys.stderr)
            print("3. The region matches the endpoint's region", file=sys.stderr)
            print("4. The credentials are properly formatted (no extra spaces or newlines)", file=sys.stderr)
            print("5. The request headers match the canonical string", file=sys.stderr)
        elif error_code == 'ModelError':
            print("\nModel Error Details:", file=sys.stderr)
            print("This usually means the request format doesn't match what the model expects.", file=sys.stderr)
            print(f"Request payload was: {payload_json}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error calling SageMaker endpoint: {str(e)}", file=sys.stderr)
        raise

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