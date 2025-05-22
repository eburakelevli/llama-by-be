from transformers import AutoModelForCausalLM, GemmaTokenizer
import torch
import json
import os

def model_fn(model_dir, context=None):
    # Load tokenizer and model with Gemma-specific settings
    tokenizer = GemmaTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,  # Use float16 for better performance
        device_map="auto",  # Automatically handle device placement
        ignore_mismatched_sizes=True  # Handle any size mismatches
    )
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # Get the input text
    prompt = input_data.get("inputs", "")
    
    # Tokenize input with Gemma-specific settings
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    # Generate response with Gemma-specific parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Gemma-specific parameters
            sliding_window=512,  # From Gemma config
            use_cache=True
        )
    
    # Decode and clean up response
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Return in the format expected by the web chat
    return {"generated_text": generated_text}

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}") 