# 🤖 AI Communication Coach

An AI-powered application that acts as a personal public speaking advisor. Upload a video of yourself speaking, and the AI will perform a holistic analysis of your communication skills, providing a comprehensive report with actionable feedback to help you improve.

 <!-- Replace with a screenshot of your Gradio app! -->

## ✨ Features

- **Multimodal Analysis:** The coach doesn't just listen to your words; it analyzes three key channels of communication:
    1.  **Visual Analysis:** Detects facial expressions (happy, sad, neutral, etc.) to gauge emotional delivery.
    2.  **Vocal Analysis:** Analyzes vocal tone (happy, sad, neutral, etc.) to understand the emotional undertone of your speech, independent of the words.
    3.  **Textual Analysis:** Transcribes your speech and analyzes pacing (Words Per Minute), use of filler words, and overall sentiment.
- **Actionable Coaching:** The system synthesizes all the data to generate personalized tips. It highlights strengths and identifies areas for improvement, such as mismatches between your words and your expression (Emotional Congruence).
- **Interactive Dashboard:** A user-friendly web interface built with Gradio allows you to easily upload videos and view your results, including a visual graph of your emotional distribution and a detailed report.

## 🛠️ Tech Stack & Architecture

This project orchestrates several state-of-the-art open-source models and libraries to achieve its comprehensive analysis.

- **Backend:** Python
- **AI / Machine Learning:**
    - **Speech-to-Text:** `openai-whisper` for highly accurate transcription.
    - **Facial Emotion Recognition:** `deepface` library leveraging deep convolutional neural networks.
    - **Vocal Emotion Recognition:** `superb/wav2vec2-base-superb-er` from Hugging Face for analyzing the tone of voice.
    - **Text Sentiment Analysis:** `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face for sentiment classification.
- **Data & Media Processing:**
    - `moviepy` for audio extraction from video files.
    - `opencv-python` for video frame processing.
    - `librosa` for audio analysis.
- **Frontend & Visualization:**
    - `Gradio` for creating the interactive web interface.
    - `Plotly` for generating the facial emotion distribution graph.

### System Workflow
1.  **Video Upload:** A user uploads a video file via the Gradio interface.
2.  **Audio Extraction:** The system uses `moviepy` to separate the audio track from the video.
3.  **Parallel Analysis:**
    - The video stream is processed by `DeepFace` to analyze facial expressions frame-by-frame.
    - The audio stream is sent to the `Wav2Vec2` model for vocal tone analysis.
    - The audio stream is also sent to `OpenAI Whisper` for transcription. The resulting text is then analyzed for pace, fillers, and sentiment.
4.  **Synthesis & Coaching:** A custom Python module takes all the structured data from the analysis phase and generates actionable coaching tips, focusing on strengths, weaknesses, and **emotional congruence**.
5.  **Report Generation:** The results are presented back to the user in a clean, tabbed interface containing coaching tips, a visual plot, and a detailed text report.

## 🚀 Getting Started

You can run this project locally or deploy it to a platform like Hugging Face Spaces.

### Prerequisites
- Python 3.8+
- `ffmpeg` (often required by `moviepy`). On macOS: `brew install ffmpeg`. On Debian/Ubuntu: `sudo apt-get install ffmpeg`.

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AI-Communication-Coach.git
    cd AI-Communication-Coach
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    Open your web browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

### Deployment

This application is ready for deployment on [Hugging Face Spaces](https://huggingface.co/spaces). Simply create a new Gradio Space and upload the `app.py` and `requirements.txt` files.

## 🎯 Future Improvements

- [ ] **Real-time Analysis:** Implement webcam support for live feedback during practice sessions.
- [ ] **Body Language Analysis:** Extend analysis beyond faces to include posture and hand gestures.
- [ ] **Advanced Textual Insights:** Incorporate topic modeling or keyword extraction to give feedback on content structure.
- [ ] **User Accounts & History:** Allow users to track their progress over time.

## 🤝 Acknowledgements

This project was made possible by the incredible work of the open-source community. Special thanks to:
- OpenAI for the Whisper model.
- The creators of the DeepFace library.
- Hugging Face for democratizing access to transformer models.
- The Gradio team for making AI interfaces so accessible.

