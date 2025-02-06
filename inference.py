import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from pypinyin import pinyin, Style
import json

class PronunciationAssessmentModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_dim, output_dim, hidden_dim=512, lstm_layers=2):
        super(PronunciationAssessmentModel, self).__init__()
        
        # Character embedding layer
        self.char_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        
        # Embedding processing BLSTM
        self.char_blstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=lstm_layers,
                                  batch_first=True, bidirectional=True)
        
        # Audio embedding linear transformation (dimensionality reduction)
        self.audio_fc = nn.Linear(input_dim, hidden_dim)
        # Audio processing BLSTM
        self.audio_blstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=lstm_layers,
                                   batch_first=True, bidirectional=True)
        
        # GAP and linear layers for both streams
        self.gap_char = nn.AdaptiveAvgPool1d(1)
        self.gap_audio = nn.AdaptiveAvgPool1d(1)
        self.fc_char = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_audio = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Final scoring module
        self.fc_final = nn.Linear(hidden_dim + 1, output_dim)

    def forward(self, audio_features, char_inputs, audio_duration):
        # Process character inputs through embedding and BLSTM
        char_embedded = self.char_embeddings(char_inputs)
        char_output, _ = self.char_blstm(char_embedded)
        char_output = self.gap_char(char_output.transpose(1, 2)).squeeze(2)
        char_output = self.fc_char(char_output)
        
        # Process audio features through linear layer and BLSTM
        audio_output = self.audio_fc(audio_features)
        audio_output, _ = self.audio_blstm(audio_output)
        audio_output = self.fc_audio(audio_output)
        
        # Concatenate both outputs
        combined_output = torch.cat((char_output, audio_output), dim=1)
        
        # Include the duration feature (ensure audio_duration is appropriately reshaped)
        # audio_duration should have shape [batch_size, 1]
        combined_output = torch.cat((combined_output, audio_duration.unsqueeze(1)), dim=1)

        # Final scoring prediction
        scores = self.fc_final(combined_output)
        return scores


def chinese_to_pinyin(chinese_text):
    pinyin_output = pinyin(chinese_text, style=Style.TONE3)
    # Join all pinyin characters into a single string with spaces
    return ' '.join([''.join(x) for x in pinyin_output])

def process_inputs(audio_features, text_indices, audio_duration):
    # Convert to tensors and ensure batch dimension
    audio_features = torch.tensor(audio_features, dtype=torch.float)
    print("shape of audio_features", audio_features.shape)
    if audio_features.ndim == 1:
        audio_features = audio_features.unsqueeze(0)  # Shape: [1, seq_len, feature_dim]

    text_indices = torch.tensor(text_indices, dtype=torch.long)
    print("shape of char_inputs", text_indices.shape)
    if text_indices.ndim == 1:
        text_indices = text_indices.unsqueeze(0)  # Shape: [1, seq_len]

    audio_duration = torch.tensor(audio_duration, dtype=torch.float)
    print("shape of audio_duration", audio_duration.shape)
    if audio_duration.ndim == 0:
        audio_duration = audio_duration.unsqueeze(0)  # Shape: [1]

    return audio_features, text_indices, audio_duration


def load_ompal_model(audio_features, chinese_text, audio_duration):
    # Convert the chinese characters to indices
    pinyin_output = chinese_to_pinyin(chinese_text)
    char_to_index_path = "char_to_index.json"
    with open(char_to_index_path, "r") as f:
        char_to_index = json.load(f)
    # Convert to indices
    text_indices = [char_to_index.get(char, 0) for char in pinyin_output]
    
    # Use in load_ompal_model
    audio_features, text_indices, audio_duration = process_inputs(
        audio_features, text_indices, audio_duration
    )

    # Initialize the model
    num_embeddings = 32  
    embedding_dim = 128  
    input_dim = 1024
    output_dim = 3
    model = PronunciationAssessmentModel(num_embeddings, embedding_dim, input_dim, output_dim)
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    st.write("OMPAL Model loaded successfully!")
    model.to(device)
    model.eval()

    # Gather predictions and actual values
    predictions = []

    # Assuming test_loader is defined and provides (features,
    with torch.no_grad():
        audio_features, text_indices, audio_duration = (
            audio_features.to(device), text_indices.to(device),
            audio_duration.to(device)
            )
        outputs = model(audio_features, text_indices, audio_duration)
        predictions.append(outputs.cpu().numpy())
    return predictions