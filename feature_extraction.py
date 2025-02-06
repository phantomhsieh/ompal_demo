import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import streamlit as st

def load_model_and_processor():
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    st.write("Pretrained Model loaded successfully!")
    return model, processor, device

def extract_features(audio_file, processor, model, device):
    features = []
    try:
        speech_array, sampling_rate = torchaudio.load(audio_file)
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)
        inputs = processor(speech_array.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values
        inputs = inputs.to(device)
        st.write("Keep going")
        with torch.no_grad():
            model.eval()
            outputs = model(inputs, output_hidden_states=True)
            feature = outputs.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()
            features.append(feature)
    except Exception as e:
        print(f"Failed to process {audio_file}: {e}")
    return features

