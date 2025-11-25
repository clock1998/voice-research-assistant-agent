from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import FileResponse
from response_generation import generate_response
from text_to_speech import synthesize_speech
from transcribe_audio import transcribe_audio
import gradio as gr
import tempfile
import os

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes)
    print(user_text)
    generated_text = generate_response(user_text)
    print(generated_text)
    response_audio = synthesize_speech(generated_text)
    return Response(content=response_audio, media_type="audio/wav")

# Gradio interface function
def chat_interface(audio_file):
    if audio_file is None:
        return None
    
    # Read the audio file
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    
    # Process through your pipeline
    user_text = transcribe_audio(audio_bytes)
    print(f"User said: {user_text}")
    
    generated_text = generate_response(user_text)
    print(f"Generated: {generated_text}")
    
    # Synthesize speech
    audio_bytes = synthesize_speech(generated_text)
    
    # Save to temporary file for Gradio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    # Return the audio file path
    return tmp_path

# Create Gradio interface
demo = gr.Interface(
    fn=chat_interface,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs=gr.Audio(label="Response Audio"),
    title="Voice Chat Assistant",
    description="Upload an audio file to chat with the AI assistant"
)

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)