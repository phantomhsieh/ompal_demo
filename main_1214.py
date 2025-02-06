import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import time
import torch
from feature_extraction import load_model_and_processor, extract_features
from inference import load_ompal_model

# Title
st.title("OMPAL - Inference API")

# Initialize session states
if 'text_submitted' not in st.session_state:
    st.session_state['text_submitted'] = False
if 'submitted_text' not in st.session_state:
    st.session_state['submitted_text'] = ""
if 'audio_option' not in st.session_state:
    st.session_state['audio_option'] = None
if 'audio_duration' not in st.session_state:
    st.session_state['audio_duration'] = None
if 'audio_file' not in st.session_state:
    st.session_state['audio_file'] = None

# Text Input
st.subheader("Enter Text")
if not st.session_state['text_submitted']:
    text_input = st.text_area("Type your chinese text here:", "")
    if st.button("Submit Text"):
        if text_input.strip():
            st.session_state['text_submitted'] = True
            st.session_state['submitted_text'] = text_input.strip()
            st.success("Text submitted successfully!")
        else:
            st.error("Text input cannot be empty.")
else:
    st.write(f"Submitted Text: {st.session_state['submitted_text']}")

# Audio Input Options
st.subheader("Audio Input")
if st.session_state['audio_option'] is None:
    st.write("Choose how to provide audio input:")
    if st.button("Record Audio"):
        st.session_state['audio_option'] = "record"
    elif st.button("Upload Audio"):
        st.session_state['audio_option'] = "upload"

# Record Audio
if st.session_state['audio_option'] == "record":
    def save_audio(audio_data, duration, filename="recorded_audio.wav", fs=44100):
        """Save the recorded audio to a .wav file."""
        samples = int(fs * duration)
        cropped_audio = audio_data[:samples]
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(fs)
            wf.writeframes((cropped_audio * 32767).astype(np.int16).tobytes())
        st.session_state['audio_file'] = filename

    if 'recording_started' not in st.session_state:
        st.session_state['recording_started'] = False
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = None
    if 'audio_data' not in st.session_state:
        st.session_state['audio_data'] = None

    col1, col2 = st.columns(2)

    if not st.session_state['recording_started']:
        if col1.button("Start Recording"):
            st.session_state['recording_started'] = True
            st.session_state['start_time'] = time.time()
            st.write("Recording... Speak now.")
            st.session_state['audio_data'] = sd.rec(int(10 * 44100), samplerate=44100, channels=1)
            if col2.button("Stop Recording"):
                sd.stop()
    else:
        if col2.button("Stop Recording"):
            sd.stop()
            st.session_state['recording_started'] = False
            duration = min(time.time() - st.session_state['start_time'], 10)
            st.session_state['audio_duration'] = duration
            save_audio(st.session_state['audio_data'], duration)
            st.success("Recording saved.")
            st.audio("recorded_audio.wav", format="audio/wav")

# Upload Audio
elif st.session_state['audio_option'] == "upload":
    uploaded_audio = st.file_uploader("Upload a recorded audio (optional):", type=["wav"])
    if uploaded_audio:
        try:
            with open("uploaded_audio.wav", "wb") as f:
                f.write(uploaded_audio.getbuffer())
            st.session_state['audio_file'] = "uploaded_audio.wav"
            with wave.open("uploaded_audio.wav", 'rb') as wf:
                st.session_state['audio_duration'] = wf.getnframes() / wf.getframerate()
            st.audio("uploaded_audio.wav", format="audio/wav")
            st.success(f"Audio uploaded successfully. Duration: {st.session_state['audio_duration']:.2f} seconds.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Load Model and Evaluate
if st.session_state['audio_file'] and st.session_state['submitted_text']:
    st.subheader("Model Evaluation")
    if st.button("Evaluate Utterance"):
        st.write("Processing the utterance...")

        # Check CUDA availability
        if torch.cuda.is_available():
            st.write("Using GPU for inference.")
        else:
            st.write("Using CPU for inference.")

        try:
            model, processor, device = load_model_and_processor()
            audio_features = extract_features(st.session_state['audio_file'], processor, model, device)

            if audio_features:
                predictions = load_ompal_model(audio_features, st.session_state['submitted_text'], st.session_state['audio_duration'])
                if predictions and isinstance(predictions[0], (list, np.ndarray)):
                    accuracy, fluency, prosody = predictions[0][0]
                    st.success(f"Results - Accuracy: {accuracy:.2f}, Fluency: {fluency:.2f}, Prosody: {prosody:.2f}")
                else:
                    st.error("Invalid predictions format.")
            else:
                st.error("Feature extraction failed.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
