#!/usr/bin/env python3
"""
Simple local chat interface using Hugging Face Transformers.
Supports conversational chat with any compatible causal language model (e.g., LLaMA 4).
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

def main():
    parser = argparse.ArgumentParser(description="Chat with a local Hugging Face model via Transformers")
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
    args = parser.parse_args()

    # token for private repo access (None means use huggingface-cli login cache)
    hf_token = args.token if args.token else None
    # Determine auth token: use explicit token or CLI login token
    auth_token = hf_token if hf_token else True

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
    print(f"Loading model '{args.model}' on {device}..." )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
            use_auth_token=auth_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_auth_token=auth_token
        )
        # Move model to appropriate device (mps or cpu)
        if device in ("cpu", "mps"):
            model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    eos_token_id = tokenizer.eos_token_id
    history = []
    print("Chat session started. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        history.append((user_input, None))
        # build prompt from history
        prompt = ""
        for u, a in history:
            prompt += f"User: {u}\n"
            if a:
                prompt += f"Assistant: {a}\n"
        prompt += "Assistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=eos_token_id
            )
        gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if "\nUser:" in text:
            text = text.split("\nUser:")[0].strip()

        history[-1] = (user_input, text)
        print(f"Assistant: {text}")

if __name__ == "__main__":
    main()