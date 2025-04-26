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

def build_prompt(history):
    prompt = ""
    for user, assistant in history:
        prompt += f"User: {user}\n"
        if assistant:
            prompt += f"Assistant: {assistant}\n"
    prompt += "Assistant:"
    return prompt

def respond(message, history, tokenizer, model, device, max_new_tokens, temperature, top_p):
    history = history or []
    # append placeholder for assistant
    history.append((message, None))
    prompt = build_prompt(history)
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
        user_input = gr.Textbox(show_label=False, placeholder="Type your message and press enter")
        # define callback
        def gr_respond(message, history):
            return respond(
                message, history, tokenizer, model, device,
                args.max_new_tokens, args.temperature, args.top_p
            )
        user_input.submit(gr_respond, [user_input, chatbot], [chatbot, user_input])
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()