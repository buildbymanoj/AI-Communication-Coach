# 🤖 AI Communication Coach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Made with Gradio](https://img.shields.io/badge/Made%20with-Gradio-orange)](https://gradio.app)
[![Powered by Hugging Face](https://img.shields.io/badge/🤗-Powered%20by%20Hugging%20Face-blue.svg)](https://huggingface.co)

An AI-powered application that acts as a personal public speaking advisor. Upload a video of yourself speaking, and the AI will perform a holistic analysis of your communication skills, providing a comprehensive report with actionable feedback to help you improve.

---

## ✨ Features

- **Multimodal Analysis:** The coach analyzes three key channels of communication for a complete picture:
    - **🎭 Visual Analysis:** Detects facial expressions (happy, sad, neutral, etc.) to gauge non-verbal emotional delivery.
    - **🎤 Vocal Analysis:** Analyzes vocal tone (happy, sad, neutral, etc.) to understand the emotional undertone of your speech.
    - **📝 Textual Analysis:** Transcribes speech and analyzes pacing (Words Per Minute), use of filler words (`um`, `ah`), and overall sentiment.
- **Actionable Coaching:** The system synthesizes all the data to generate personalized tips. It highlights strengths and identifies areas for improvement, such as mismatches between your words and your expression (**Emotional Congruence**).
- **Interactive Dashboard:** A user-friendly web interface built with Gradio allows for easy video uploads and presents results in a clear, tabbed format with visual graphs.

## 🛠️ Tech Stack & Architecture

This project orchestrates several state-of-the-art open-source models and libraries.

- **Backend:** **Python**
- **AI / Machine Learning:**
    - **Speech-to-Text:** `openai-whisper`
    - **Facial Emotion Recognition:** `deepface`
    - **Vocal Emotion Recognition:** `superb/wav2vec2-base-superb-er` (from Hugging Face)
    - **Text Sentiment Analysis:** `distilbert-base-uncased-finetuned-sst-2-english` (from Hugging Face)
- **Data & Media Processing:** `moviepy`, `opencv-python`, `librosa`
- **Frontend & Visualization:** `Gradio`, `Plotly`

### System Workflow

The application follows a clear data processing pipeline, visually represented below:

```mermaid
graph TD
    A[🎥 Video Upload] --> B{Extract Audio};
    B --> C[🎧 Audio File];
    A --> D[🎞️ Video Frames];

    subgraph "Parallel Analysis"
        D --> E[👨‍💻 DeepFace: Facial Emotion Analysis];
        C --> F[🗣️ Wav2Vec2: Vocal Tone Analysis];
        C --> G[📝 Whisper: Speech-to-Text];
    end

    G --> H[🧐 Text Analysis: Pace, Fillers, Sentiment];

    subgraph "Synthesis"
        E --> I{🧠 Coaching Engine};
        F --> I;
        H --> I;
    end

    I --> J[📊 Generate Final Report: Tips, Plot, Transcript];
    J --> K[🖥️ Display in Gradio Interface];
