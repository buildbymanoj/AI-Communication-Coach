# --- 1. SETUP AND INSTALLATIONS ---
# We run this first to make sure all libraries are ready.
print("⏳ Installing dependencies... This is only needed once per session.")
!pip install -q gradio
!pip install -q deepface
!pip install -q transformers torch
!pip install -q moviepy
!pip install -q librosa
!pip install -q soundfile
!pip install -q openai-whisper
!pip install -q datasets
!pip install -q plotly #<-- ### NEW ### For creating graphs

print("✅ Dependencies installed.")


# --- 2. IMPORTS AND MODEL LOADING ---
import gradio as gr
import cv2
from deepface import DeepFace
from collections import Counter
import whisper
import os
from moviepy.editor import VideoFileClip
import librosa
from transformers import pipeline
import re
import json
import torch
import plotly.graph_objects as go 

print("🧠 Loading AI models into memory... (This may take a moment)")
# Check for GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Whisper Model
whisper_model = whisper.load_model("base")
# Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
# Load Vocal Emotion Recognition Model
vocal_emotion_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)
print("✅ AI models loaded.")


# --- 3. HELPER FUNCTIONS (The AI Brain) ---

# ### UPDATED ###
# Function from Phase 1: Visual Analysis
# Now returns a dictionary of data instead of a formatted string.
def analyze_visuals(video_path):
    """Analyzes facial expressions and returns a dictionary of the emotion distribution."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        emotions_detected = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Analyze one frame per second
            if frame_count % frame_rate == 0:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                # Check if a face was detected in the frame
                if result and isinstance(result, list) and result[0].get('dominant_emotion'):
                     emotions_detected.append(result[0]['dominant_emotion'])
            frame_count += 1
        cap.release()

        if not emotions_detected:
            return None # Return None if no faces were ever detected

        emotion_distribution = Counter(emotions_detected)
        total_detections = sum(emotion_distribution.values())

        # Normalize the distribution to percentages
        emotion_percentages = {emotion: (count / total_detections) for emotion, count in emotion_distribution.items()}

        # Find the dominant emotion across the entire video
        dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)

        return {
            "emotion_distribution": emotion_percentages,
            "dominant_emotion": dominant_emotion
        }
    except Exception as e:
        print(f"Error in visual analysis: {e}")
        return None

# ### UPDATED ###
# Function for Vocal Tone Analysis
# Now returns a dictionary of data instead of a formatted string.
def analyze_vocal_tone(audio_path):
    """Analyzes vocal emotion and returns a dictionary of results."""
    try:
        predictions = vocal_emotion_classifier(audio_path, top_k=4)
        emotion_mapping = {'ang': 'Angry', 'hap': 'Happy', 'neu': 'Neutral', 'sad': 'Sad'}
        
        # Structure the results
        vocal_results = {
            "dominant_vocal_emotion": emotion_mapping.get(predictions[0]['label'], predictions[0]['label']),
            "confidence": predictions[0]['score'],
            "all_predictions": {emotion_mapping.get(p['label'], p['label']): p['score'] for p in predictions}
        }
        return vocal_results
    except Exception as e:
        print(f"Error in vocal tone analysis: {e}")
        return None

# ### UPDATED ###
# Function for Audio & Text Analysis
# Now returns a dictionary of all data points.
def analyze_audio_and_text(video_path):
    """
    Extracts audio, transcribes, and analyzes vocal tone and text content.
    Returns a dictionary containing all analysis data.
    """
    audio_path = "temp_audio.wav"
    analysis_data = {}
    try:
        with VideoFileClip(video_path) as video_clip:
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return {"error": f"Could not extract audio: {e}"}

    # Vocal Tone Analysis
    vocal_data = analyze_vocal_tone(audio_path)
    if vocal_data:
        analysis_data.update(vocal_data)

    # Transcription and Text Analysis
    try:
        audio_duration_sec = librosa.get_duration(path=audio_path)
        analysis_data['duration'] = audio_duration_sec

        transcription_result = whisper_model.transcribe(audio_path, fp16=False)
        full_transcript = transcription_result['text'].strip()
        analysis_data['transcript'] = full_transcript

        if not full_transcript:
            analysis_data['transcript'] = "No speech detected."
            return analysis_data

        FILLER_WORDS = re.compile(r'\b(um|uh|ah|like|so|you know|basically|actually)\b', re.IGNORECASE)
        word_count = len(full_transcript.split())
        wpm = (word_count / audio_duration_sec) * 60 if audio_duration_sec > 0 else 0
        filler_count = len(FILLER_WORDS.findall(full_transcript))
        sentiment_result = sentiment_analyzer(full_transcript)

        analysis_data['pace_wpm'] = wpm
        analysis_data['filler_words'] = filler_count
        analysis_data['text_sentiment'] = sentiment_result[0]['label']

    except Exception as e:
        print(f"Error during audio/text processing: {e}")
        analysis_data['error'] = f"Error during audio/text processing: {e}"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return analysis_data

# --- 4. NEW COACHING & VISUALIZATION FUNCTIONS ---

### NEW ###
def generate_coaching_tips(visual_data, audio_data):
    """Generates personalized feedback based on the analysis data."""
    tips = []
    
    # --- Pace Feedback ---
    if 'pace_wpm' in audio_data:
        wpm = audio_data['pace_wpm']
        if wpm > 170:
            tips.append(f"💡 **Pacing:** Your speaking pace is quite fast at {wpm:.0f} WPM. Consider taking deliberate pauses to let key points sink in. This helps with audience comprehension.")
        elif wpm < 120 and audio_data.get('transcript') != "No speech detected.":
            tips.append(f"💡 **Pacing:** Your pace is {wpm:.0f} WPM, which is a bit slow. Try to inject more energy into your delivery to keep your audience engaged.")
        else:
            tips.append(f"✅ **Pacing:** Your pace of {wpm:.0f} WPM is within the ideal range (120-170 WPM). Great job!")

    # --- Filler Word Feedback ---
    if 'filler_words' in audio_data and 'duration' in audio_data:
        fillers = audio_data['filler_words']
        duration_min = audio_data['duration'] / 60
        fillers_per_min = fillers / duration_min if duration_min > 0 else 0
        if fillers_per_min > 4:
            tips.append(f"💡 **Clarity:** You used {fillers} filler words (e.g., um, like, so). Try practicing your script or pausing silently instead of using fillers to sound more confident.")
        else:
            tips.append("✅ **Clarity:** You used very few filler words. This makes you sound confident and clear.")

    # --- Emotional Congruence Feedback (The most advanced part) ---
    if visual_data and audio_data:
        facial_emotion = visual_data.get('dominant_emotion')
        vocal_emotion = audio_data.get('dominant_vocal_emotion')
        text_sentiment = audio_data.get('text_sentiment', '').upper()

        # Check for mismatch between words and face/voice
        if text_sentiment == 'POSITIVE' and (facial_emotion in ['sad', 'angry'] or vocal_emotion in ['Sad', 'Angry']):
            tips.append(f"💡 **Emotional Alignment:** Your words seem positive, but your dominant facial expression was '{facial_emotion}' and your vocal tone sounded '{vocal_emotion}'. Try to align your non-verbal cues with your message for greater impact. Smile when sharing good news!")
        elif text_sentiment == 'NEGATIVE' and (facial_emotion == 'happy' or vocal_emotion == 'Happy'):
            tips.append(f"💡 **Emotional Alignment:** Your words seem serious, but your dominant expression was 'happy'. Ensure your delivery matches the gravity of your topic to maintain credibility.")
        
        # General emotional feedback
        if facial_emotion == 'neutral' and visual_data['emotion_distribution'].get('neutral', 0) > 0.7:
             tips.append("💡 **Expression:** Your expression was neutral over 70% of the time. Don't be afraid to be more expressive! Using facial expressions that match your words helps connect with your audience.")
        elif facial_emotion in ['sad', 'angry']:
             tips.append(f"💡 **Expression:** Your dominant expression was '{facial_emotion}'. If this wasn't intentional for a serious topic, be mindful that it can make you seem unapproachable or upset.")
        else:
             tips.append("✅ **Emotional Alignment:** Your facial expressions and vocal tone appear to be well-aligned with your message. This builds trust and connection.")

    if not tips:
        return "No specific tips to generate. The analysis might have been incomplete."
        
    return "\n\n".join(tips)

### NEW ###
def create_emotion_plot(visual_data):
    """Creates a Plotly bar chart from the emotion distribution data."""
    if not visual_data or "emotion_distribution" not in visual_data:
        return None # Return None if no data to plot

    data = visual_data["emotion_distribution"]
    emotions = [e.capitalize() for e in data.keys()]
    percentages = [p * 100 for p in data.values()]

    fig = go.Figure([go.Bar(x=emotions, y=percentages, text=[f'{p:.1f}%' for p in percentages], textposition='auto')])
    fig.update_layout(
        title_text='Facial Emotion Distribution',
        xaxis_title="Emotion",
        yaxis_title="Percentage (%)",
        yaxis_range=[0,100],
        template="plotly_white"
    )
    return fig


# --- 5. MAIN ORCHESTRATOR FUNCTION ---
# ### UPDATED ###
# Now returns multiple, structured outputs for the new Gradio interface.
def the_ai_communication_coach(video_path):
    print(f"🚀 Starting full analysis for: {video_path}")
    
    # Run all analyses and get structured data
    visual_data = analyze_visuals(video_path)
    audio_text_data = analyze_audio_and_text(video_path)
    
    # --- Generate Outputs ---
    
    # 1. Coaching Tips
    print("🧠 Generating coaching tips...")
    coaching_report = generate_coaching_tips(visual_data, audio_text_data)
    
    # 2. Emotion Plot
    print("📊 Creating emotion plot...")
    emotion_plot = create_emotion_plot(visual_data)

    # 3. Detailed Text Report
    print("📝 Assembling detailed report...")
    # Visual Part
    if visual_data:
        visual_report = "--- Visual Emotion Report ---\n"
        visual_report += f"Dominant Facial Emotion: {visual_data['dominant_emotion'].capitalize()}\n"
        visual_report += "Distribution of detected emotions:\n"
        for emotion, percentage in visual_data['emotion_distribution'].items():
            visual_report += f"- {emotion.capitalize()}: {percentage:.1%}\n"
    else:
        visual_report = "--- Visual Emotion Report ---\nNo faces were detected or an error occurred."

    # Audio/Text Part
    audio_text_report = "--- Audio & Textual Report ---\n"
    if 'error' in audio_text_data:
        audio_text_report += audio_text_data['error']
    else:
        if 'dominant_vocal_emotion' in audio_text_data:
            audio_text_report += f"Dominant Vocal Emotion: {audio_text_data['dominant_vocal_emotion']} (Confidence: {audio_text_data.get('confidence', 0):.1%})\n\n"
        
        audio_text_report += "Textual Content Analysis:\n"
        audio_text_report += f"- Pace: {audio_text_data.get('pace_wpm', 0):.2f} Words Per Minute\n"
        audio_text_report += f"- Filler Words Count: {audio_text_data.get('filler_words', 0)}\n"
        audio_text_report += f"- Overall Sentiment of Words: {audio_text_data.get('text_sentiment', 'N/A')}\n\n"
        audio_text_report += "Full Transcript:\n"
        audio_text_report += f'"{audio_text_data.get("transcript", "Transcript not available.")}"'

    detailed_report = f"{visual_report}\n\n----------------------------------------------------------\n\n{audio_text_report}"
    
    print("✅ Full analysis complete!")
    
    # Return all the pieces for the UI, in order
    return coaching_report, emotion_plot, detailed_report


# --- 6. GRADIO INTERFACE ---
# ### UPDATED ###
# A more advanced interface with Tabs, a Plot, and multiple Textboxes.
print("🚀🚀🚀 Launching the AI Communication Coach! 🚀🚀🚀")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 AI-Powered Communication Coach 🤖
        Upload a video of yourself speaking. The AI will analyze your **facial expressions**, **vocal tone**, and **speech patterns**
        to provide a complete feedback report with actionable tips.
        *Analysis may take a few minutes for longer videos.*
        """
    )
    with gr.Row():
        video_input = gr.Video(label="Upload Your Presentation Video")
    
    analyze_button = gr.Button("Analyze My Communication", variant="primary")

    with gr.Tabs():
        with gr.TabItem("⭐ Key Feedback & Coaching"):
            with gr.Row():
                coaching_output = gr.Textbox(label="Actionable Coaching Tips", lines=15, interactive=False)
                plot_output = gr.Plot(label="Facial Emotion Distribution")
        with gr.TabItem("📄 Detailed Report"):
            detailed_report_output = gr.Textbox(label="Full Analysis Breakdown", lines=20, interactive=False)
    
    analyze_button.click(
        fn=the_ai_communication_coach,
        inputs=video_input,
        outputs=[coaching_output, plot_output, detailed_report_output]
    )

demo.launch(debug=True, share=True)
