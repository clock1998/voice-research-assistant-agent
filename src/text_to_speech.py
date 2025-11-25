from TTS.api import TTS
import numpy as np
import io
import soundfile as sf
import torch

def synthesize_speech(text):
    """
    Synthesize speech from text using Coqui TTS.
    Returns audio data as bytes (WAV format).
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize TTS model
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)
    
    # Generate speech (returns numpy array)
    audio_array = tts.tts(text=text)
    
    # Get sample rate (default to 22050 if not available)
    sample_rate = getattr(tts.synthesizer, 'output_sample_rate', 22050)
    if sample_rate is None:
        sample_rate = 22050
    
    # Convert numpy array to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='WAV')
    audio_bytes = buffer.getvalue()
    
    return audio_bytes
    