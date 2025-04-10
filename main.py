import streamlit as st
import numpy as np
import wave
import os
from streamlit_option_menu import option_menu
import time
import sounddevice as sd  # Ensure you have this module installed
import requests 
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av  # Audio/Video processing
import io

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

# Send audio to API
def send_audio_to_api(audio_file_path):
    api_endpoint = "http://140.112.41.120:8000/upload-audio/"  # 確保這是正確的 API 端點
    headers = {"access_token": "your-secret-api-key"}  # 添加 API 密鑰到請求頭
    with open(audio_file_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(api_endpoint, files=files, headers=headers)
        if response.status_code == 200:
            result = response.json()
            #st.success(f"Audio duration from API: {result['duration_seconds']} seconds.")
            # Delete the audio file after processing
            os.remove(audio_file_path)
            return result
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            st.error("Failed to upload audio to API.")

# Audio Recording Component
def audio_recorder():
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        ),
    )

    if webrtc_ctx.audio_receiver:
        st.info("Recording... Speak now!")
        audio_frames = []
        
        # Collect audio chunks
        for frame in webrtc_ctx.audio_receiver.get_frames(timeout=10):
            audio_frames.append(frame.to_ndarray())
        
        if audio_frames:
            # Convert to playable audio
            audio_data = np.concatenate(audio_frames)
            audio_bytes = io.BytesIO()
            AudioSegment(
                audio_data.tobytes(),
                frame_rate=44100,
                sample_width=audio_data.dtype.itemsize,
                channels=1
            ).export(audio_bytes, format="wav")
            
            st.audio(audio_bytes, format="audio/wav")
            return audio_bytes.getvalue()
    return None

def save_audio(audio_data, duration, filename="recorded_audio.wav", fs=44100):
    """Save the recorded audio to a .wav file with precise duration handling."""
    try:
        # Get actual recorded samples (not just the buffer size)
        actual_samples = int(fs * duration)
        cropped_audio = audio_data[:actual_samples, 0]  # Take first channel if stereo
        
        # Remove DC offset and normalize
        cropped_audio = cropped_audio - np.mean(cropped_audio)
        max_val = np.max(np.abs(cropped_audio))
        if max_val > 0:
            cropped_audio = cropped_audio / max_val
        
        # Apply fade-out to prevent clicks
        fade_samples = min(512, actual_samples)  # 512 samples (~12ms at 44100Hz)
        if fade_samples > 0:
            fade_window = np.linspace(1.0, 0.0, fade_samples)
            cropped_audio[-fade_samples:] *= fade_window
        
        # Convert to 16-bit PCM
        int_audio = (cropped_audio * 32767).astype(np.int16)
        
        # Save to WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(int_audio.tobytes())
        
        return filename
        
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None


# Home Page
if selected == "Home":
    st.subheader("線上華語發音評分系統\nOnline Mandarin Pronunciation Scoring System")

    # Initialize session states
    if 'text_submitted' not in st.session_state:
        st.session_state['text_submitted'] = False
    if 'submitted_text' not in st.session_state:
        st.session_state['submitted_text'] = ""
    if 'audio_duration' not in st.session_state:
        st.session_state['audio_duration'] = None
    if 'audio_file' not in st.session_state:
        st.session_state['audio_file'] = None

    # Audio Input Options
    st.subheader("Audio Input")
    input_method = st.radio("Select input method:", 
                          ("Record Audio", "Upload Audio"),
                          index=1)

    # Record Audio
    if input_method == "Record Audio":
        audio_bytes = audio_recorder()
        if audio_bytes:
            # Save to file
            with open("recorded_audio.wav", "wb") as f:
                f.write(audio_bytes)
            
            st.session_state['audio_file'] = "recorded_audio.wav"
            st.success("Recording saved!")
        '''if 'recording_started' not in st.session_state:
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

                # Wait for recording to fully stop
                time.sleep(0.1)

                duration = min(time.time() - st.session_state['start_time'], 10)
                st.session_state['audio_duration'] = duration
                st.session_state['audio_file'] = save_audio(st.session_state['audio_data'], duration)
                st.success("Recording saved.")
                st.audio("recorded_audio.wav", format="audio/wav")'''

    # Upload Audio
    elif input_method == "Upload Audio":
        uploaded_audio = st.file_uploader("Upload a recorded audio:", type=["wav"])
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
    st.subheader("Enter Transcript")
    if not st.session_state['text_submitted']:
        text_input = st.text_area("Please provide the transcript (the Chinese text matching the audio recording) here:", "")
        if st.button("Submit and run evaluation"):
            if text_input.strip():
                st.session_state['text_submitted'] = True
                st.session_state['submitted_text'] = text_input.strip()
                st.success("Text submitted successfully!")
            else:
                st.error("Text input cannot be empty.")
    else:
        st.write(f"Submitted Text: {st.session_state['submitted_text']}")

    # Inference Section
    if st.session_state['audio_file'] and st.session_state['submitted_text']:
        st.subheader("Model Evaluation")
        with st.spinner("Processing..."):
            result = send_audio_to_api(st.session_state['audio_file'])
        
        if result:
            # Display the results
            st.success(f"Accuracy: {result['results']['accuracy']:.2f}, Fluency: {result['results']['fluency']:.2f}, Prosody: {result['results']['prosody']:.2f}")
            # Display the message
            st.success("You can refresh the page (F5) to re-evaluate.")


    # Footer Information
    st.markdown("---")
    st.markdown("""
    本語料庫為國科會計畫「以深度學習為基礎之聲調辨識系統建置」（計畫編號：NSC 112-2410-H-002-061-MY2）之研究成果。
    感謝江振宇教授和葉丙成教授於計畫執行期間給予之寶貴建議。

    音檔為法國國立東方語文學院漢學系學生之錄音，感謝劉展岳和劉芸菁協助收集音檔。

    計畫主持人：劉德馨  \n
    協同研究人員：謝文崴 r11942078@ntu.edu.tw \n
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
