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
        params['use_auth_token'] = hf_token
    print(f"Loading model '{model_name}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, **({'use_auth_token': hf_token} if hf_token else {}))
    model = AutoModelForCausalLM.from_pretrained(model_name, **params)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model

def build_prompt(history, file_store=None):
    prompt = ""
    if file_store:
        for name, text in file_store:
            prompt += f"[Content from {name}]\n{text}\n\n"
    for user, assistant in history:
        prompt += f"User: {user}\n"
        if assistant:
            prompt += f"Assistant: {assistant}\n"
    prompt += "Assistant:"
    return prompt

def respond(message, history, tokenizer, model, device, max_new_tokens, temperature, top_p, file_store=None):
    history = history or []
    # append placeholder for assistant
    history.append((message, None))
    # include any uploaded file context in the prompt
    prompt = build_prompt(history, file_store)
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
    # extract generated tokens
    gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    # remove any trailing user prompts
    if "\nUser:" in text:
        text = text.split("\nUser:")[0].strip()
    # update history
    history[-1] = (message, text)
    return history, ""

def main():
    parser = argparse.ArgumentParser(description="Web chat with a local Hugging Face model via Gradio")
    parser.add_argument("-m", "--model", required=True,
                        help="Model ID or local path (e.g., meta-llama/Llama-4-7b-chat)")
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        help="Device to run on (default: cuda if available else cpu)")
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = args.token if args.token else None

    tokenizer, model = load_model(args.model, device, hf_token)

    # launch Gradio interface
    with gr.Blocks() as demo:
        # display the loaded model name as the chat title
        chatbot = gr.Chatbot(label=args.model)
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
                        history.append((f"Error reading CSV {file_path}", str(e)))
                        continue
                elif ext == ".pdf":
                    try:
                        reader = PdfReader(file_path)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                    except Exception as e:
                        history.append((f"Error reading PDF {file_path}", str(e)))
                        continue
                else:
                    continue
                store.append((os.path.basename(file_path), text))
                history.append((f"Uploaded {os.path.basename(file_path)}", ""))
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