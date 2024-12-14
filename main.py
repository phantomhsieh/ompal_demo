import streamlit as st
import numpy as np
import wave
import torch
from feature_extraction import load_model_and_processor, extract_features
from inference import load_ompal_model

# Title
st.title("OMPAL - Inference API")

# Initialize session states
if 'state' not in st.session_state:
    st.session_state['state'] = {
        'text_submitted': False,
        'submitted_text': "",
        'audio_option': None,
        'audio_duration': None,
        'audio_file': None
    }

state = st.session_state['state']

# Text Input
st.subheader("Enter Text")
if not state['text_submitted']:
    text_input = st.text_area("Type your Chinese text here:", "")
    if st.button("Submit Text"):
        if text_input.strip():
            state['text_submitted'] = True
            state['submitted_text'] = text_input.strip()
            st.success("Text submitted successfully!")
        else:
            st.error("Text input cannot be empty.")
else:
    st.write(f"Submitted Text: {state['submitted_text']}")

# Upload Audio
st.subheader("Audio Input")
uploaded_audio = st.file_uploader("Upload a recorded audio (optional):", type=["wav"])
if uploaded_audio:
    try:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_audio.getbuffer())
        state['audio_file'] = "uploaded_audio.wav"
        with wave.open("uploaded_audio.wav", 'rb') as wf:
            state['audio_duration'] = wf.getnframes() / wf.getframerate()
        st.audio("uploaded_audio.wav", format="audio/wav")
        st.success(f"Audio uploaded successfully. Duration: {state['audio_duration']:.2f} seconds.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Load Model and Evaluate
if state['audio_file'] and state['submitted_text']:
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
            audio_features = extract_features(state['audio_file'], processor, model, device)

            if audio_features:
                predictions = load_ompal_model(audio_features, state['submitted_text'], state['audio_duration'])
                if predictions and isinstance(predictions[0], (list, np.ndarray)):
                    accuracy, fluency, prosody = predictions[0][0]
                    st.success(f"Results - Accuracy: {accuracy:.2f}, Fluency: {fluency:.2f}, Prosody: {prosody:.2f}")
                else:
                    st.error("Invalid predictions format.")
            else:
                st.error("Feature extraction failed.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
