from transformers import AutoModelForCausalLM, GemmaTokenizer, GemmaForCausalLM, AutoConfig
import torch
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir, context=None):
    try:
        # Load config first to get model-specific settings
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        logger.info(f"Loaded model config: {config.model_type}")
        
        # Load tokenizer with specific settings
        tokenizer = GemmaTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            padding_side="left",  # Important for generation
            model_max_length=config.max_position_embeddings
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
            torch_dtype=torch.float16 if config.torch_dtype == "float32" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            use_cache=config.use_cache
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
        
        # Get the input text
        prompt = input_data.get("inputs", "")
        if not prompt:
            return {"generated_text": "Error: Empty input prompt"}
        
        # Get generation parameters from input or use defaults
        max_new_tokens = input_data.get("max_new_tokens", 256)
        temperature = input_data.get("temperature", 0.7)
        top_p = input_data.get("top_p", 0.9)
        
        # Tokenize input with proper padding and truncation
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_position_embeddings - max_new_tokens
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
                sliding_window=config.sliding_window,
                use_cache=config.use_cache,
                attention_mode="sliding_chunks" if config.sliding_window_pattern else "global"
            )
        
        # Decode and clean up response
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
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