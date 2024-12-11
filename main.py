import streamlit as st
import numpy as np
import wave

# Title
st.title("OMPAL - Inference API")

# Text input
st.subheader("Enter Text")
text_input = st.text_area("Type your text here:", "")

# Audio recording setup
st.subheader("Record Audio (less than 10 seconds)")
