#!/usr/bin/env python3
"""
Web-based chat interface to interact with a local Hugging Face causal language model using Gradio.
Supports conversational chat with any compatible model (e.g., LLaMA 4).
"""
import sys
import argparse

try:
    import torch
except ImportError:
    print("Error: 'torch' library not found. Install with 'pip install torch'", file=sys.stderr)
    sys.exit(1)
import os
import pandas as pd
from PyPDF2 import PdfReader
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Error: 'transformers' library not found. Install with 'pip install transformers accelerate sentencepiece'", file=sys.stderr)
    sys.exit(1)
try:
    import gradio as gr
except ImportError:
    print("Error: 'gradio' library not found. Install with 'pip install gradio'", file=sys.stderr)
    sys.exit(1)

def load_model(model_name, device, hf_token=None):
    params = dict(
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    if hf_token:
        # use the new 'token' parameter (use_auth_token is deprecated)
        params['token'] = hf_token
    print(f"Loading model '{model_name}' on {device}...")
    # pass HF token using 'token' (use_auth_token deprecated)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        **({'token': hf_token} if hf_token else {})
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **params)
    # Move model to appropriate device (mps or cpu)
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
    # history is a list of dicts with 'role' and 'content'
    history = history or []
    # include the new user message in prompt history
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
    # update history with user message and assistant response
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": text}
    ]
    return new_history, ""

def main():
    parser = argparse.ArgumentParser(description="Web chat with a local Hugging Face model via Gradio")
    parser.add_argument("-m", "--model", required=True,
                        help="Model ID or local path (e.g., meta-llama/Llama-4-7b-chat)")
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
    args = parser.parse_args()

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
        # define callback including file context
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