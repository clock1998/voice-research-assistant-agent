import torch
from transformers import pipeline
conversation_history = []

llm = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.1-8B",
    model_kwargs={"dtype": torch.bfloat16}, 
    device_map="auto"
)

def generate_response(user_text):
    # Add user message to history (chat template expects "content" not "text")
    conversation_history.append({"role": "user", "content": user_text})
    
    # Generate response with temperature=0 for deterministic output
    outputs = llm(user_text, max_new_tokens=100, return_full_text=False, temperature=1)
    bot_response = outputs[0]["generated_text"].strip()
    
    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": bot_response})
    return bot_response