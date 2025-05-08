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
# Google Sign-In dependencies
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as grequests
    from fastapi import Request, HTTPException
except ImportError:
    print("Error: Missing dependencies for Google Sign-In. Install with 'pip install google-auth fastapi'", file=sys.stderr)
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
    args = parser.parse_args()

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
    # Ensure Google Client ID is set
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not google_client_id:
        print("Error: Please set the GOOGLE_CLIENT_ID environment variable", file=sys.stderr)
        sys.exit(1)

    # launch Gradio interface with Google Sign-In
    # Inject Google Identity Services scripts into the page <head>
    google_head = f"""
<script src="https://accounts.google.com/gsi/client" onload="initGoogleSignIn()"></script>
<script>
function handleCredentialResponse(response) {{
    const id_token = response.credential;
    fetch('/api/auth/google', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ idToken: id_token }}) }})
        .then(res => {{
            if (res.ok) {{
                document.getElementById('login-container').style.display = 'none';
                document.getElementById('chat-container').style.display = 'block';
            }} else {{
                res.text().then(text => {{ alert('Google sign-in failed: ' + text); }});
            }}
        }});
}}
function initGoogleSignIn() {{
    google.accounts.id.initialize({{ client_id: "{google_client_id}", callback: handleCredentialResponse }});
    google.accounts.id.renderButton(
        document.getElementById("google-signin-button"),
        {{ theme: "outline", size: "large" }}
    );
}}
</script>
"""
    with gr.Blocks(head=google_head) as demo:
        # Google Sign-In endpoint
        @demo.app.post("/api/auth/google")
        async def auth_google(request: Request):
            data = await request.json()
            token = data.get("idToken")
            if not token:
                raise HTTPException(status_code=400, detail="Missing ID token")
            try:
                idinfo = id_token.verify_oauth2_token(token, grequests.Request(), google_client_id)
            except ValueError:
                raise HTTPException(status_code=401, detail="Invalid ID token")
            return {"status": "success", "email": idinfo.get("email"), "name": idinfo.get("name"), "picture": idinfo.get("picture")}

        # Login container (visible by default)
        with gr.Column(visible=True, elem_id="login-container"):
            gr.HTML(value="<h2 style='text-align:center;'>Please sign in with Google to continue</h2><div id='google-signin-button' style='text-align:center; margin-top:20px;'></div>")

        # Chat container (hidden until sign-in)
        with gr.Column(visible=False, elem_id="chat-container"):
            chatbot = gr.Chatbot(label=args.model, type='messages')
            file_store = gr.State([])
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
                    history.append({"role": "assistant", "content": f"Uploaded {basename}"})
                return history, store, None

            file_input.upload(process_file, [file_input, chatbot, file_store], [chatbot, file_store, file_input])

            user_input = gr.Textbox(show_label=False, placeholder="Type your message and press enter")
            def gr_respond(message, history, store):
                return respond(
                    message, history, tokenizer, model, device,
                    args.max_new_tokens, args.temperature, args.top_p, store
                )
            user_input.submit(gr_respond, [user_input, chatbot, file_store], [chatbot, user_input])

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()