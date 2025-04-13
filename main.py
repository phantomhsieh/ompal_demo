import streamlit as st
import numpy as np
import wave
import os
from streamlit_option_menu import option_menu
import time
import sounddevice as sd  # Ensure you have this module installed
import requests 
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av  # Audio/Video processing
import io
import queue
import asyncio

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 3. ASYNCIO WORKAROUND (ADD THIS BLOCK RIGHT HERE)
import asyncio
import nest_asyncio

def fix_asyncio_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)

fix_asyncio_event_loop()  # Call it immediately

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
    """Browser-based audio recorder using WebRTC with improved reliability"""
    # Initialize session state for recording control
    if 'recording_in_progress' not in st.session_state:
        st.session_state.recording_in_progress = False
    if 'recording_complete' not in st.session_state:
        st.session_state.recording_complete = False

    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,  # Increased buffer size
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
        media_stream_constraints={
            "audio": {
                "sampleRate": 44100,  # Explicit sample rate
                "channelCount": 1,     # Mono audio
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True
            },
            "video": False
        },
    )

    if webrtc_ctx.audio_receiver and not st.session_state.recording_complete:
        st.session_state.recording_in_progress = True
        status_text = st.empty()
        status_text.info("Recording... Speak now!")
        
        audio_frames = []
        silent_frames = 0
        max_silent_frames = 10  # Allow some silence before stopping
        
        try:
            while st.session_state.recording_in_progress:
                frame = webrtc_ctx.audio_receiver.get_frame(timeout=2)  # Shorter timeout
                frame_data = frame.to_ndarray()
                
                # Validate audio frame
                if frame_data.size > 0:
                    rms = np.sqrt(np.mean(frame_data**2))  # Calculate RMS volume
                    if rms > 0.01:  # Threshold for non-silent audio
                        audio_frames.append(frame_data)
                        silent_frames = 0
                    else:
                        silent_frames += 1
                        
                    # Stop if too many silent frames
                    if silent_frames >= max_silent_frames:
                        st.warning("Stopping recording due to silence")
                        break
        except queue.Empty:
            st.warning("Recording timed out")
        except Exception as e:
            st.error(f"Recording error: {str(e)}")
            return None
        
        st.session_state.recording_in_progress = False
        st.session_state.recording_complete = True
        
        if audio_frames:
            status_text.success("Recording complete! Processing...")
            try:
                audio_data = np.concatenate(audio_frames)
                
                # Create audio segment with correct parameters
                audio_segment = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=44100,
                    sample_width=audio_data.dtype.itemsize,
                    channels=1
                )
                
                # Normalize volume
                audio_segment = audio_segment.normalize()
                
                # Export to WAV
                audio_bytes = io.BytesIO()
                audio_segment.export(audio_bytes, format="wav")
                
                status_text.empty()
                return audio_bytes.getvalue()
            except Exception as e:
                st.error(f"Audio processing error: {str(e)}")
                return None
        else:
            status_text.error("No valid audio recorded")
            return None
    
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

def clamp_and_feedback(result):
    # Clamp scores between 0 and 5
    clamped_scores = {
        'accuracy': max(0, min(5, result['results']['accuracy'])),
        'fluency': max(0, min(5, result['results']['fluency'])),
        'prosody': max(0, min(5, result['results']['prosody']))
    }
    
    # Calculate average score
    avg_score = (clamped_scores['accuracy'] + clamped_scores['fluency'] + clamped_scores['prosody']) / 3
    
    # Determine feedback based on average score (all under 10 words)
    if avg_score < 3:
        feedback = "Needs significant improvement. Keep practicing!"
    elif 3 <= avg_score < 4:
        feedback = "Good effort! Some areas need refinement."
    elif 4 <= avg_score < 4.5:
        feedback = "Great job! You're close to excellent."
    else:  # 4.5-5
        feedback = "Excellent performance! Nearly flawless."
    
    # Display the results with feedback
    st.success(
        f"Accuracy: {clamped_scores['accuracy']:.2f}, "
        f"Fluency: {clamped_scores['fluency']:.2f}, "
        f"Prosody: {clamped_scores['prosody']:.2f}\n"
        "(The rating scale is from 0 to 5.)"   
    )
    st.success(f"Feedback: {feedback}")
    st.success("You can refresh the page (F5) to re-evaluate.")

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
        st.text("Maximum audio length: ~1 sentence (under 15 sec).")
        
        # Under maintenance
        st.warning("Recording feature is under maintenance. Please use the upload option.")

        # Skip the recording part for now
        #audio_bytes = audio_recorder()
        #if audio_bytes:
            # Save to file
        #    with open("recorded_audio.wav", "wb") as f:
        #        f.write(audio_bytes)
            
        #    st.session_state['audio_file'] = "recorded_audio.wav"
        #    st.audio(audio_bytes, format="audio/wav")
        #    st.success("Recording saved!")
  
    # Upload Audio
    elif input_method == "Upload Audio":
        st.text("Maximum audio length: ~1 sentence (under 15 sec).")
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
            clamp_and_feedback(result)


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
