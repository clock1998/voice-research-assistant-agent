from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# Load the core components
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

def synthesize_speech(text):
    
    inputs = processor(text=text, return_tensors="pt")
    # Load a dataset containing pre-calculated X-vectors (speaker embeddings)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    # Select a specific speaker (e.g., index 7306 is for the 'slt' female voice)
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # Generate the speech waveform
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    return speech
    