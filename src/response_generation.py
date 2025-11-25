import torch
from transformers import pipeline
conversation_history = []

# Auto-detect device
if torch.cuda.is_available():
    device = 0  # Use first CUDA device
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = -1  # CPU

llm = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response