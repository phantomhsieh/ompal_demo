# OMPAL: Open‑Source Mandarin Pronunciation Assessment Corpus

A Streamlit‑based web app for recording/uploading Mandarin speech, sending it to a scoring API, and displaying pronunciation feedback (accuracy, fluency, prosody).

---

## Features

- **Home**  
  - Upload or (future) record a short Mandarin audio clip  
  - Enter matching Chinese transcript  
  - Send to remote API for scoring (0–5 scale)  
  - Receive real‑time feedback and suggested improvements  

- **Audio Files**  
  - Browse and play curated audio samples by proficiency level  
  - View corresponding Chinese text for each sample  

---

## Installation

1. **Clone repo**  
   ```bash
   git clone https://github.com/your‑org/ompal.git
   cd ompal
   ```

2. **Create & activate a virtualenv**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Place assets**  
   - `logo.png` in project root  
   - Under `audio_files/`, create `beginner/` and `intermediate/` folders with subfolders of samples and matching `.txt` files.

---

## Configuration

- **API endpoint**  
  Edit `api_endpoint` in `send_audio_to_api()` if your scoring server differs.
- **API key**  
  Set `access_token` header in `send_audio_to_api()` or use an environment variable.

---

## Usage

```bash
streamlit run app.py
```

1. Open the URL shown in your browser (usually http://localhost:8501).  
2. Navigate via the side menu: **Home** or **Audio Files**.  
3. On **Home**:  
   - Upload a WAV file (≤15 s)  
   - Enter the matching Chinese text  
   - Click **Submit and run evaluation**  
   - View scores & feedback  
4. On **Audio Files**:  
   - Select level, folder, and sample  
   - Play audio and read the transcript  

---

## Project Structure

```
├── app.py
├── requirements.txt
├── packages.txt
├── logo.png
├── audio_files/
│   ├── beginner/
│   │   ├── lesson1/
│   │   │   ├── sample1.wav
│   │   │   └── lesson1.txt
│   │   └── …
│   └── intermediate/
│       └── …
└── README.md
```

---

## Acknowledgments

This work was supported by MOST Project NSC 112‑2410‑H‑002‑061‑MY2.  
Special thanks to Prof. Jiang Zhen‑Yu, Prof. Yeh Bing‑Cheng, and student assistants Liu Zhan‑Yue & Liu Yun‑Jing.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
```