import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import logging

# Disable logging for a clean chat experience
logging.getLogger("transformers").setLevel(logging.ERROR)

def chat():
    model_path = "./unfettered-Qwen2.5-0.5B-Instruct"
    print(f"Loading ablated model from {model_path}...")
    
    try:
        from transformers import BitsAndBytesConfig
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Proper config for 4-bit loading
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quant_config
        )

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nModel loaded. Type 'quit' to exit.\n")
    
    while True:
        prompt = input("User: ")
        if prompt.lower() in ["quit", "exit"]:
            break
            
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("Assistant: ", end="", flush=True)
        
        print("(Thinking... CPU inference takes time)", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response.strip() + "\n")

if __name__ == "__main__":
    chat()
