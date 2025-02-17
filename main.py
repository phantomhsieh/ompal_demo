import streamlit as st
import numpy as np
import wave
import torch
import os
from feature_extraction import load_model_and_processor, extract_features
from inference import load_ompal_model
from streamlit_option_menu import option_menu
import time
import sounddevice as sd  # Ensure you have this module installed

# Center the title and subtitle using HTML
st.markdown(
    """
    <div style="text-align: center;">
        <h1>OMPAL</h1>
    </div>
    <h2>Open-Source Mandarin Pronunciation Assessment Corpus for Global Learners</h2>
    """,
    unsafe_allow_html=True,
)
# Logo Display
logo_path = "logo.png"  # Ensure this file is in the same directory as the script

# Display the image
st.image(logo_path, use_container_width=True)

# Define pages
selected = option_menu(
    menu_title="Navigation",  # Title for the menu
    options=["Home", "Audio Files"],  # Menu options
    icons=["house", "file-music"],  # Icons for the options
    menu_icon="menu-button-wide",  # Icon for the menu
    default_index=0,  # Default selected index
    orientation="vertical",  # Menu orientation
)

# Home Page
if selected == "Home":
    st.subheader("線上華語發音評分系統（維護中請勿使用）\nOnline Mandarin Pronunciation Scoring System")

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

    # Text Input
    st.subheader("Enter Text")
    if not st.session_state['text_submitted']:
        text_input = st.text_area("Type your Chinese text here:", "")
        if st.button("Submit Text"):
            if text_input.strip():
                st.session_state['text_submitted'] = True
                st.session_state['submitted_text'] = text_input.strip()
                st.success("Text submitted successfully!")
            else:
                st.error("Text input cannot be empty.")
    else:
        st.write(f"Submitted Text: {st.session_state['submitted_text']}")

    # Inference Section (Currently Under Maintenance)
    if st.session_state['audio_file'] and st.session_state['submitted_text']:
        st.subheader("Model Evaluation (Under Maintenance)")
        st.info("This feature is currently under maintenance. Please check back later.")

    # Footer Information
    st.markdown("---")
    st.markdown("""
    本語料庫為國科會計畫「以深度學習為基礎之聲調辨識系統建置」（計畫編號：NSC 112-2410-H-002-061-MY2）之研究成果。
    感謝江振宇教授和葉丙成教授於計畫執行期間給予之寶貴建議。

    音檔為法國國立東方語文學院漢學系學生之錄音，感謝劉展岳和劉芸菁協助收集音檔。

    計畫主持人：劉德馨  
    協同研究人員：謝文崴  
    計畫助理：廖芳婷、李沛欣、楊元嘉
    """)

# Audio Files Page
elif selected == "Audio Files":

    # Base directories for levels
    base_audio_dir = {
        "Beginner level": "audio_files/beginner",
        "Intermediate level": "audio_files/intermediate"
    }

    # Page title
    st.subheader("Audio Menu")

    # Step 1: Select level
    level = st.selectbox("Select Level", list(base_audio_dir.keys()), key="level")

    # Check if the base directory exists
    base_dir = base_audio_dir[level]
    if not os.path.exists(base_dir):
        st.error(f"Base directory '{base_dir}' not found. Please create it and add your folders.")
    else:
        # Step 2: Select folder within the level
        folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
        if folders:
            folder = st.selectbox("Select Folder", folders, key="folder")
            selected_folder_path = os.path.join(base_dir, folder)
            
            # Step 3: List audio files in the selected folder
            audio_files = sorted([f for f in os.listdir(selected_folder_path) if f.endswith((".wav", ".mp3"))])

            if audio_files:
                # Step 4: Select an audio file to play
                audio_file = st.selectbox("Select Audio File", audio_files, key="audio_file")
                if selected_folder_path is None or audio_file is None:
                    st.error("Please select an audio file.")
                else:
                    audio_path = os.path.join(selected_folder_path, audio_file)
                    # Display the selected audio file
                    st.audio(audio_path, format="audio/wav" if audio_file.endswith(".wav") else "audio/mp3")
                    st.write(f"Audio File: {audio_file}")
            else:
                st.warning(f"No audio files found in the folder '{folder}'.")

            # Step 5: Display folder-specific text content
            text_file = f"{folder}.txt"
            text_path = os.path.join(selected_folder_path, text_file)
            st.subheader("Text Content")

            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    folder_text_content = f.read()
                st.text(folder_text_content)
            else:
                st.warning(f"No text file found for folder '{folder}'. Expected file: {text_file}")

        else:
            st.warning(f"No folders found in '{base_dir}'. Please create folders with audio files.")
