from transformers import AutoModelForCausalLM, GemmaTokenizer, GemmaForCausalLM, AutoConfig
import torch
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_prompt_for_gemma(prompt):
    """
    Format the prompt according to Gemma's chat template.
    Gemma expects a specific format for chat interactions.
    """
    # Remove any existing Gemma formatting to avoid double-formatting
    prompt = prompt.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
    
    # Split into turns if it contains "User:" or "Assistant:"
    turns = []
    current_role = None
    current_content = []
    
    for line in prompt.split("\n"):
        if line.startswith("User:"):
            if current_role and current_content:
                turns.append((current_role, "\n".join(current_content)))
            current_role = "user"
            current_content = [line[5:].strip()]
        elif line.startswith("Assistant:"):
            if current_role and current_content:
                turns.append((current_role, "\n".join(current_content)))
            current_role = "assistant"
            current_content = [line[10:].strip()]
        elif current_role:
            current_content.append(line.strip())
    
    if current_role and current_content:
        turns.append((current_role, "\n".join(current_content)))
    
    # Format according to Gemma's chat template
    formatted_prompt = ""
    for role, content in turns:
        formatted_prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    
    # Add final assistant turn if not present
    if not formatted_prompt.endswith("<start_of_turn>assistant\n"):
        formatted_prompt += "<start_of_turn>assistant\n"
    
    return formatted_prompt

def model_fn(model_dir, context=None):
    try:
        # Load config first to get model-specific settings
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        logger.info(f"Loaded model config: {config.model_type}")
        
        # Load tokenizer with specific settings for Gemma
        tokenizer = GemmaTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            padding_side="left",  # Important for generation
            model_max_length=config.max_position_embeddings,
            add_eos_token=True,  # Ensure EOS token is added
            add_bos_token=True   # Ensure BOS token is added
        )
        
        # Ensure tokenizer has the correct special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = config.bos_token_id
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        
        # Load model with specific settings from config
        model = GemmaForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.float16,  # Gemma models are optimized for float16
            device_map="auto",
            trust_remote_code=True,
            use_cache=True  # Enable KV cache for better performance
        )
        
        # Ensure model is in eval mode
        model.eval()
        logger.info("Model and tokenizer loaded successfully")
        
        return {"model": model, "tokenizer": tokenizer, "config": config}
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            # Handle both 'inputs' and 'gemma3_text' formats
            if "gemma3_text" in input_data:
                input_data["inputs"] = input_data["gemma3_text"]
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {str(e)}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    try:
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        config = model_dict["config"]
        
        # Get the input text and format it for Gemma
        prompt = input_data.get("inputs", "")
        if not prompt:
            return {"generated_text": "Error: Empty input prompt"}
        
        # Format the prompt according to Gemma's chat template
        formatted_prompt = format_prompt_for_gemma(prompt)
        
        # Get generation parameters from input or use defaults
        max_new_tokens = input_data.get("max_new_tokens", 256)
        temperature = input_data.get("temperature", 0.7)
        top_p = input_data.get("top_p", 0.9)
        
        # Tokenize input with proper padding and truncation
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_position_embeddings - max_new_tokens,
            add_special_tokens=True  # Ensure special tokens are added
        ).to(model.device)
        
        # Generate response with model-specific parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                # Gemma-specific parameters
                sliding_window=config.sliding_window if hasattr(config, 'sliding_window') else None,
                use_cache=True,
                attention_mode="sliding_chunks" if hasattr(config, 'sliding_window_pattern') and config.sliding_window_pattern else "global",
                repetition_penalty=1.1,  # Slight repetition penalty for better responses
                length_penalty=1.0,      # Neutral length penalty
                no_repeat_ngram_size=3   # Prevent repetition of 3-grams
            )
        
        # Decode and clean up response
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Remove any remaining Gemma formatting tokens
        generated_text = generated_text.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
        
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"generated_text": f"Error during generation: {str(e)}"}

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        try:
            return json.dumps(prediction)
        except Exception as e:
            logger.error(f"Error in output_fn: {str(e)}")
            return json.dumps({"error": str(e)})
    raise ValueError(f"Unsupported content type: {response_content_type}") 