import streamlit as st
import numpy as np
import wave
import sounddevice as sd
from queue import Queue
import threading

# Global variables for audio recording
recording_flag = False
audio_queue = Queue()
sample_rate = 44100
audio_data = []

def audio_callback(indata, frames, time, status):
    """Called for each audio block from microphone."""
    if recording_flag:
        audio_queue.put(indata.copy())

def start_recording():
    global recording_flag, audio_data
    recording_flag = True
    audio_data = []
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=audio_callback,
        dtype='float32'
    )
    stream.start()
    return stream

def stop_recording(stream):
    global recording_flag
    recording_flag = False
    stream.stop()
    stream.close()
    
    # Combine all audio blocks
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
    
    if audio_data:
        return np.concatenate(audio_data)
    return None

# Streamlit UI
st.title("Audio Recorder with Stop Button")

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'audio_ready' not in st.session_state:
    st.session_state.audio_ready = False

col1, col2 = st.columns(2)

# Start/Stop buttons
if not st.session_state.recording:
    if col1.button("🎤 Start Recording"):
        st.session_state.recording = True
        st.session_state.stream = start_recording()
        st.session_state.audio_ready = False
        st.rerun()  # Use st.rerun() instead of experimental_rerun()

if st.session_state.recording:
    if col2.button("⏹️ Stop Recording"):
        audio_array = stop_recording(st.session_state.stream)
        st.session_state.recording = False
        
        if audio_array is not None:
            # Save as WAV file
            with wave.open("recording.wav", 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_array * 32767).astype(np.int16).tobytes())
            
            st.session_state.audio_ready = True
            st.session_state.audio_file = "recording.wav"
            st.session_state.audio_duration = len(audio_array) / sample_rate
            st.rerun()

# Display audio and submit button
if st.session_state.get('audio_ready', False):
    st.audio(st.session_state.audio_file, format='audio/wav')
    st.success(f"Recorded {st.session_state.audio_duration:.2f} seconds")
    
    if st.button("Submit to API"):
        # Replace with your actual API call
        # result = send_audio_to_api(st.session_state.audio_file)
        st.success("Audio submitted successfully!")
        st.balloons()