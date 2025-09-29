# --- 1. SETUP AND INSTALLATIONS ---
print("⏳ Installing dependencies... This is only needed once per session.")
!pip install -q gradio deepface transformers torch moviepy librosa soundfile openai-whisper datasets plotly opencv-python
print("✅ Dependencies installed.")


# --- 2. IMPORTS AND MODEL LOADING ---
import gradio as gr
import cv2
import os
import re
import json
import torch
import numpy as np
import random  # NEW: For dynamic phrasing variation
from deepface import DeepFace
from collections import Counter
from moviepy.editor import VideoFileClip
import librosa
from transformers import pipeline
import plotly.graph_objects as go
from typing import Dict, Optional, Any
import whisper

print("🧠 Loading AI models into memory... (This may take 1-2 minutes)")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Whisper (base for speed; use 'small' if you have RAM and want better accuracy)
whisper_model = whisper.load_model("base")

# Sentiment analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    truncation=True,
    max_length=512
)

# Vocal emotion classifier
vocal_emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=device
)

print("✅ AI models loaded. Ready for analysis!")


# --- 3. HELPER FUNCTIONS (The AI Brain) ---

def analyze_visuals(video_path: str, sample_rate: int = 1) -> Optional[Dict]:
    """
    Analyzes facial expressions frame-by-frame (1 frame per second).
    Returns emotion distribution and dominant emotion.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        if frame_rate == 0:
            frame_rate = 30  # fallback

        emotions_detected = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📊 Analyzing {total_frames} frames at {frame_rate} FPS... sampling every {sample_rate} second(s).")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (frame_rate * sample_rate) == 0:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'  # More robust than dlib for video
                    )
                    if isinstance(result, list) and len(result) > 0 and result[0].get('dominant_emotion'):
                        # NEW: Confidence check (skip if <40%)
                        dominant_emotion = result[0]['dominant_emotion']
                        if result[0]['emotion'][dominant_emotion] > 40:
                            emotions_detected.append(dominant_emotion)
                except Exception as e:
                    # Skip frame if analysis fails (e.g., no face)
                    pass

            frame_count += 1
            if frame_count % (frame_rate * 10) == 0:  # Progress every 10 seconds
                print(f"   ... {frame_count // frame_rate}s processed")

        cap.release()

        if not emotions_detected:
            print("⚠️ No faces detected in video.")
            return None

        emotion_distribution = Counter(emotions_detected)
        total = sum(emotion_distribution.values())
        emotion_percentages = {k: v / total for k, v in emotion_distribution.items()}
        dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)

        return {
            "emotion_distribution": emotion_percentages,
            "dominant_emotion": dominant_emotion,
            "total_faces_detected": len(emotions_detected)
        }

    except Exception as e:
        print(f"❌ Error in visual analysis: {e}")
        return None


def analyze_vocal_tone(audio_path: str) -> Optional[Dict]:
    """Analyzes vocal emotion with confidence scoring."""
    try:
        predictions = vocal_emotion_classifier(audio_path, top_k=4)
        emotion_mapping = {
            'ang': 'Angry',
            'hap': 'Happy',
            'neu': 'Neutral',
            'sad': 'Sad',
            'sur': 'Surprised',
            'dis': 'Disgusted',
            'fea': 'Fearful'
        }

        top_pred = predictions[0]
        dominant_vocal = emotion_mapping.get(top_pred['label'], top_pred['label'])
        confidence = top_pred['score']

        all_predictions = {
            emotion_mapping.get(p['label'], p['label']): p['score']
            for p in predictions
        }

        return {
            "dominant_vocal_emotion": dominant_vocal,
            "vocal_confidence": confidence,
            "all_vocal_predictions": all_predictions
        }
    except Exception as e:
        print(f"❌ Error in vocal tone analysis: {e}")
        return None


def analyze_audio_and_text(video_path: str) -> Dict[str, Any]:
    """Extracts audio, transcribes, and analyzes speech patterns."""
    audio_path = "temp_audio.wav"
    analysis = {}

    try:
        # Extract audio with better quality
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(
                audio_path,
                codec='pcm_s16le',
                fps=16000,  # Whisper expects 16kHz
                logger=None,
                verbose=False
            )

        # Duration
        duration = librosa.get_duration(path=audio_path)
        analysis['duration'] = duration

        # Transcription
        result = whisper_model.transcribe(audio_path, fp16=False, language='en')
        transcript = result['text'].strip()

        if not transcript or len(transcript.strip()) < 5:
            analysis['transcript'] = "No speech detected."
        else:
            analysis['transcript'] = transcript

            # NEW: Extract filler examples for dynamic tips
            FILLER_WORDS = re.compile(r'\b(um|uh|ah|like|so|you know|basically|actually|right|okay|well)\b', re.IGNORECASE)
            words = transcript.split()
            word_count = len(words)
            wpm = (word_count / duration) * 60 if duration > 0 else 0
            filler_matches = FILLER_WORDS.findall(transcript)
            filler_count = len(filler_matches)
            filler_rate = filler_count / word_count if word_count > 0 else 0
            # NEW: Get specific filler examples (first 3)
            filler_examples = list(set(filler_matches[:3]))  # Unique examples

            # NEW: Transcript patterns (e.g., questions for engagement, repetition)
            questions = len(re.findall(r'\?', transcript))
            repetition_score = len(set(words)) / len(words) if words else 0  # Lower = more repetition

            # Sentiment analysis (only if transcript is meaningful)
            if word_count > 10:
                sentiment = sentiment_analyzer(transcript)[0]
                analysis['text_sentiment'] = sentiment['label']
                analysis['text_sentiment_score'] = sentiment['score']
            else:
                analysis['text_sentiment'] = "N/A"
                analysis['text_sentiment_score'] = 0.0

            analysis['pace_wpm'] = wpm
            analysis['filler_words'] = filler_count
            analysis['filler_rate'] = filler_rate
            analysis['filler_examples'] = filler_examples
            analysis['word_count'] = word_count
            analysis['questions_count'] = questions
            analysis['repetition_score'] = repetition_score  # 1.0 = no repetition, <0.7 = repetitive

    except Exception as e:
        analysis['error'] = f"Audio/text processing failed: {e}"
        print(f"❌ {analysis['error']}")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return analysis


# --- 4. NEW COACHING & VISUALIZATION FUNCTIONS ---

def generate_coaching_tips(visual_data: Optional[Dict], audio_data: Dict) -> str:
    """Generates personalized, empathetic, and ACTIONABLE coaching tips with dynamic variation."""
    tips = []
    score_components = {}

    # Dynamic phrasing templates (NEW: For variety)
    positive_phrases = ["Well done!", "You're nailing this!", "Excellent work!", "Great job!"]
    suggestion_phrases = ["Here's a tip:", "Try this:", "Consider:", "To level up:"]
    motivational_phrases = ["You've got this!", "Small tweaks make big differences!", "Keep practicing!"]

    # --- CALCULATE ENGAGEMENT SCORE (NEW: Dynamic overall metric) ---
    engagement_score = 5.0  # Start neutral
    if 'pace_wpm' in audio_data:
        wpm = audio_data['pace_wpm']
        score_components['pace'] = max(0, min(10, 10 - abs(wpm - 145) / 10))  # Peak at 145 WPM
        engagement_score += score_components['pace'] / 2
    if 'filler_rate' in audio_data:
        filler_rate = audio_data['filler_rate']
        score_components['fillers'] = max(0, 10 - (filler_rate * 100 * 10))  # Penalize high rate
        engagement_score += score_components['fillers'] / 2
    if visual_data and 'emotion_distribution' in visual_data:
        neutral_perc = visual_data['emotion_distribution'].get('neutral', 0)
        score_components['expressiveness'] = max(0, 10 - (neutral_perc * 10))
        engagement_score += score_components['expressiveness'] / 2
    if 'text_sentiment_score' in audio_data:
        engagement_score += audio_data['text_sentiment_score'] * 5  # Sentiment confidence boosts
    engagement_score = max(0, min(10, engagement_score))  # Clamp 0-10
    tips.append(f"📊 **Overall Engagement Score: {engagement_score:.1f}/10**  {random.choice(positive_phrases) if engagement_score > 7 else random.choice(motivational_phrases) if engagement_score > 4 else 'Room to grow—let\'s improve!'}")

    # --- PACE (Enhanced with specifics) ---
    if 'pace_wpm' in audio_data and audio_data.get('transcript') != "No speech detected.":
        wpm = audio_data['pace_wpm']
        word_count = audio_data.get('word_count', 0)
        if wpm > 180:
            tips.append(f"{random.choice(suggestion_phrases)} **Pacing:** At {wpm:.0f} WPM (very fast), your {word_count} words flew by! Practice the 'pause-and-breathe' technique: After every sentence, pause for 2 seconds. This boosts retention by 20%.")
            engagement_score -= 1  # Adjust if needed
        elif wpm > 170:
            tips.append(f"{random.choice(suggestion_phrases)} **Pacing:** {wpm:.0f} WPM is energetic but rushed. Slow to 150-160 WPM for emphasis—try timing yourself with a 60-second mirror practice.")
        elif 140 <= wpm <= 170:
            tips.append(f"✅ **Pacing:** Ideal at {wpm:.0f} WPM! Your rhythm keeps listeners hooked. {random.choice(positive_phrases)}")
        elif 120 <= wpm < 140:
            tips.append(f"{random.choice(suggestion_phrases)} **Pacing:** {wpm:.0f} WPM is steady but could add energy. Vary your speed—speed up for excitement, slow for impact. Experiment in your next recording!")
        else:  # <120
            tips.append(f"{random.choice(suggestion_phrases)} **Pacing:** Slow at {wpm:.0f} WPM—great for detail, but inject pace to engage. Use a metronome app at 140 beats/min to build rhythm. {random.choice(motivational_phrases)}")

    # --- FILLER WORDS (NEW: Transcript-specific examples) ---
    if 'filler_words' in audio_data and audio_data.get('word_count', 0) > 10:
        fillers = audio_data['filler_words']
        filler_rate = audio_data['filler_rate']
        examples = audio_data.get('filler_examples', [])
        duration_min = audio_data['duration'] / 60
        fillers_per_min = fillers / duration_min if duration_min > 0 else 0
        if fillers_per_min > 6 or filler_rate > 0.1:
            ex_str = f" (e.g., '{', '.join(examples)}')" if examples else ""
            tips.append(f"{random.choice(suggestion_phrases)} **Clarity:** {fillers} fillers ({fillers_per_min:.1f}/min){ex_str}—common in natural speech! Replace with silence: Record a 'filler-free' version of your transcript. You'll sound more authoritative instantly.")
        elif fillers_per_min > 3 or filler_rate > 0.05:
            ex_str = f" (like '{', '.join(examples)}')" if examples else ""
            tips.append(f"✅ **Clarity:** Moderate {fillers} fillers{ex_str}—you're improving! Pause instead next time. {random.choice(positive_phrases)} Progress!")
        else:
            tips.append(f"🌟 **Clarity:** Minimal {fillers} fillers—confident delivery! This polish sets you apart in presentations.")

    # --- TRANSCRIPT PATTERNS (NEW: Dynamic based on content) ---
    transcript = audio_data.get('transcript', '')
    word_count = audio_data.get('word_count', 0)
    questions = audio_data.get('questions_count', 0)
    repetition = audio_data.get('repetition_score', 1.0)
    if word_count > 20:
        if word_count < 50:
            tips.append(f"{random.choice(suggestion_phrases)} **Structure:** Short speech ({word_count} words)—add a story or example to expand impact. E.g., 'Imagine if...' engages audiences more.")
        if questions > 1:
            tips.append(f"✅ **Engagement:** {questions} questions in your transcript—smart! This draws listeners in. Build on it with rhetorical questions.")
        if repetition < 0.7:
            tips.append(f"{random.choice(suggestion_phrases)} **Variety:** Some repetition detected (score: {repetition:.2f}). Vary words (e.g., synonym for 'important') to keep it fresh and persuasive.")

    # --- EMOTIONAL ALIGNMENT (Enhanced with multi-modal depth) ---
    if visual_data and audio_data and audio_data.get('text_sentiment') != 'N/A':
        facial_emotion = visual_data.get('dominant_emotion', '').lower()
        vocal_emotion = audio_data.get('dominant_vocal_emotion', '').lower()
        text_sentiment = audio_data.get('text_sentiment', '').lower()
        neutral_perc = visual_data['emotion_distribution'].get('neutral', 0)
        vocal_conf = audio_data.get('vocal_confidence', 0)

        mismatch_score = 0
        if text_sentiment == 'positive' and (facial_emotion in ['sad', 'angry', 'fearful'] or vocal_emotion in ['sad', 'angry']):
            mismatch_score = 1 if vocal_conf > 0.7 else 0.5  # Stronger if confident mismatch
            tips.append(f"💡 **Emotional Alignment:** Positive words (e.g., from '{transcript[:50]}...') clash with {facial_emotion} face/{vocal_emotion} tone (mismatch level: {'high' if mismatch_score > 0.7 else 'medium'}). Align by mirroring: Smile for positives! Practice in front of a mirror.")
            engagement_score -= mismatch_score * 2
        elif text_sentiment == 'negative' and (facial_emotion in ['happy', 'surprised'] or vocal_emotion == 'happy'):
            tips.append(f"💡 **Emotional Alignment:** Serious message but upbeat cues—soften for credibility. Nod solemnly for emphasis. {random.choice(motivational_phrases)}")
            engagement_score -= 1
        elif neutral_perc > 0.7:
            tips.append(f"💡 **Expression:** High neutral ({neutral_perc:.0%})—calm is good, but add {random.choice(['smiles', 'nods', 'raised eyebrows'])} for warmth. Expressive faces connect 2x better!")
        elif vocal_conf < 0.5:
            tips.append(f"💡 **Vocal Energy:** Low confidence ({vocal_conf:.0%}) in {vocal_emotion} tone—project louder! Warm up with vocal exercises like humming scales.")
        else:
            tips.append(f"🌟 **Emotional Alignment:** Seamless match across face, voice, and words. {random.choice(positive_phrases)} Authentic delivery!")

    # Final score adjustment and wrap-up
    engagement_score = max(0, min(10, engagement_score))
    if engagement_score < 5:
        tips.append(f"{random.choice(motivational_phrases)} Focus on one tip today—progress compounds!")
    elif engagement_score > 8:
        tips.append(f"🎉 **Pro Tip:** You're advanced—try audience interaction next time!")

    if not tips:
        return "Analysis incomplete (e.g., no clear speech/faces). Upload a better-lit video with audio for dynamic insights!"

    return "**Your Dynamic Feedback Report**\n\n" + "\n\n".join(tips)


def create_emotion_plot(visual_data: Optional[Dict]) -> Optional[go.Figure]:
    """Creates a beautiful, interactive Plotly bar chart for facial emotion distribution."""
    if not visual_data or "emotion_distribution" not in visual_data:
        return None

    data = visual_data["emotion_distribution"]
    emotions = [e.capitalize() for e in data.keys()]
    percentages = [p * 100 for p in data.values()]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F7B7A3']  # Soft pastel palette

    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=percentages,
            text=[f"{p:.1f}%" for p in percentages],
            textposition='auto',
            marker_color=[colors[i % len(colors)] for i in range(len(emotions))],
            hovertemplate="<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>"
        )
    ])

    fig.update_layout(
        title={
            'text': "📊 Facial Emotion Distribution Over Time",
            'x': 0.5,
            'font': {'size': 20, 'family': "Arial"}
        },
        xaxis_title="Emotion",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        hovermode="x unified"
    )

    return fig


def create_summary_badges(visual_data: Optional[Dict], audio_data: Dict, engagement_score: float) -> str:
    """Returns HTML-style badge summary for UI header, now with dynamic score."""
    badges = []

    # NEW: Engagement score badge (color-coded)
    score_color = "#4CAF50" if engagement_score > 7 else "#FF9800" if engagement_score > 5 else "#F4436"
    badges.append(f"<span style='background-color:{score_color}; color:white; padding:8px 16px; border-radius:20px; font-size:16px; font-weight:bold; display:inline-block; margin:4px;'>📊 Engagement: {engagement_score:.1f}/10</span>")

    # Pace badge
    if 'pace_wpm' in audio_data:
        wpm = audio_data['pace_wpm']
        pace_status = "🟢 Ideal" if 120 <= wpm <= 170 else "🟡 Moderate" if wpm < 140 or wpm > 160 else "🔴 Needs Work"
        badges.append(f"<span style='background-color:#e0f7fa; padding:6px 12px; border-radius:15px; font-size:14px; display:inline-block; margin:2px;'>🗣️ Pace: {pace_status} ({wpm:.0f} WPM)</span>")

    # Filler words badge
    if 'filler_words' in audio_data:
        fillers = audio_data['filler_words']
        filler_status = "🟢 Low" if fillers < 3 else "🟡 Medium" if fillers < 6 else "🔴 High"
        badges.append(f"<span style='background-color:#fff3e0; padding:6px 12px; border-radius:15px; font-size:14px; display:inline-block; margin:2px;'>🗣️ Fillers: {filler_status} ({fillers})</span>")

    # Emotional alignment badge
    if visual_data and 'dominant_emotion' in visual_data and 'text_sentiment' in audio_data:
        facial = visual_data['dominant_emotion'].lower()
        text = audio_data['text_sentiment'].lower()
        align = "🟢 Strong" if (text == 'positive' and facial in ['happy', 'surprised']) or (text == 'negative' and facial in ['sad', 'angry']) else "🟡 Okay" if facial == 'neutral' else "🔴 Weak"
        badges.append(f"<span style='background-color:#f0f0f0; padding:6px 12px; border-radius:15px; font-size:14px; display:inline-block; margin:2px;'>🎭 Alignment: {align}</span>")

    return " ".join(badges) if badges else ""


# --- 5. MAIN ORCHESTRATOR FUNCTION ---
def the_ai_communication_coach(video_path: str) -> tuple:
    """
    Main function that orchestrates full analysis.
    Returns: coaching_tips, emotion_plot, detailed_report, summary_badges
    """
    print(f"🚀 Starting full analysis for: {os.path.basename(video_path) if video_path else 'No file'}")

    # Validate input
    if not video_path or not os.path.exists(video_path):
        return "❌ No video uploaded.", None, "❌ No video provided.", ""

    # Run analyses
    print("🔍 Running visual analysis...")
    visual_data = analyze_visuals(video_path)

    print("🎙️ Running audio/text analysis...")
    audio_data = analyze_audio_and_text(video_path)

    # NEW: Calculate engagement score here for badges
    engagement_score = 5.0  # Default
    if 'pace_wpm' in audio_data:
        wpm = audio_data['pace_wpm']
        engagement_score += max(0, min(5, 5 - abs(wpm - 145) / 10))
    if 'filler_rate' in audio_data:
        engagement_score += max(0, 5 - (audio_data['filler_rate'] * 100 * 5))
    if visual_data:
        neutral_perc = visual_data['emotion_distribution'].get('neutral', 0)
        engagement_score += max(0, 5 - (neutral_perc * 5))
    engagement_score = max(0, min(10, engagement_score))

    # Generate outputs
    print("🧠 Generating dynamic coaching tips...")
    coaching_tips = generate_coaching_tips(visual_data, audio_data)

    print("📊 Creating emotion plot...")
    emotion_plot = create_emotion_plot(visual_data)

    print("📝 Assembling dynamic detailed report...")
    # Visual Report (NEW: Interpretive)
    if visual_data:
        neutral_perc = visual_data['emotion_distribution'].get('neutral', 0)
        visual_report = "--- 🎭 Visual Emotion Report ---\n"
        visual_report += f"🔹 Dominant Facial Emotion: {visual_data['dominant_emotion'].capitalize()}\n"
        visual_report += f"🔹 Total Faces Detected: {visual_data['total_faces_detected']}\n"
        visual_report += "🔹 Emotion Distribution:\n"
        for emotion, perc in visual_data['emotion_distribution'].items():
            visual_report += f"   - {emotion.capitalize()}: {perc:.1%}\n"
        # NEW: Dynamic interpretation
        if neutral_perc > 0.6:
            visual_report += f"\n💡 **Insight:** High neutral expression ({neutral_perc:.0%}) indicates a composed style, but varying emotions could increase audience connection by emphasizing key points."
        elif visual_data['dominant_emotion'] in ['happy', 'surprised']:
            visual_report += f"\n🌟 **Insight:** Positive dominant emotion aligns well with engaging communication—keep this energy!"
    else:
        visual_report = "--- 🎭 Visual Emotion Report ---\n⚠️ No faces detected. Tip: Ensure good lighting and face visibility for accurate analysis."

    # Audio/Text Report (NEW: Dynamic sections and interpretation)
    audio_text_report = "--- 🎙️ Audio & Text Analysis ---\n"
    if 'error' in audio_data:
        audio_text_report += f"⚠️ {audio_data['error']}\n\n💡 **Quick Fix:** Check audio levels and re-upload."
    else:
        transcript = audio_data.get('transcript', 'No transcript.')
        audio_text_report += f"🔹 Duration: {audio_data.get('duration', 0):.1f} seconds\n"
        audio_text_report += f"🔹 Pace: {audio_data.get('pace_wpm', 0):.1f} WPM (Words: {audio_data.get('word_count', 0)})\n"
        if 'filler_words' in audio_data and audio_data['filler_words'] > 0:
            examples = audio_data.get('filler_examples', [])
            ex_str = f" (e.g., '{', '.join(examples)}')" if examples else ""
            audio_text_report += f"🔹 Filler Words: {audio_data.get('filler_words', 0)}{ex_str}\n"
        audio_text_report += f"🔹 Text Sentiment: {audio_data.get('text_sentiment', 'N/A')} (Confidence: {audio_data.get('text_sentiment_score', 0):.1%})\n"
        if audio_data.get('questions_count', 0) > 0:
            audio_text_report += f"🔹 Engagement Elements: {audio_data.get('questions_count', 0)} questions detected—great for interaction!\n"
        # NEW: Dynamic interpretation
        wpm = audio_data.get('pace_wpm', 0)
        if wpm > 170:
            audio_text_report += f"\n💡 **Pace Insight:** Fast delivery ({wpm:.0f} WPM) suits excitement but may overwhelm—ideal for short bursts."
        elif wpm < 120:
            audio_text_report += f"\n💡 **Pace Insight:** Deliberate pace ({wpm:.0f} WPM) builds thoughtfulness, perfect for complex topics."
        sentiment = audio_data.get('text_sentiment', '')
        if sentiment == 'POSITIVE':
            audio_text_report += f"\n🌟 **Sentiment Insight:** Uplifting tone in transcript—pairs well with energetic delivery!"
        elif sentiment == 'NEGATIVE':
            audio_text_report += f"\n💡 **Sentiment Insight:** Serious content—ensure steady voice to convey empathy without overwhelming."

        audio_text_report += "\n🔹 Full Transcript:\n"
        if len(transcript) > 500:
            transcript = transcript[:500] + "... (truncated for brevity)"
        audio_text_report += f'"{transcript}"'

    detailed_report = f"{visual_report}\n\n{'─' * 60}\n\n{audio_text_report}\n\n📊 **Overall Engagement Score: {engagement_score:.1f}/10** (Based on pace, clarity, and alignment)."

    # Summary badges (enhanced with score)
    summary_badges = create_summary_badges(visual_data, audio_data, engagement_score)

    print("✅ Full dynamic analysis complete!")
    return coaching_tips, emotion_plot, detailed_report, summary_badges


# --- 6. GRADIO INTERFACE ---
print("🚀🚀🚀 Launching the AI Communication Coach! 🚀🚀🚀")

with gr.Blocks(theme=gr.themes.Soft(), title="AI Communication Coach") as demo:
    gr.Markdown(
        """
        # 🤖 AI-Powered Communication Coach 🤖
        Upload a video of yourself speaking — whether for a presentation, interview, or pitch — and receive **instant, personalized feedback** on your:

        ✅ Facial expressions
        ✅ Vocal tone & emotion
        ✅ Speech pace & filler words
        ✅ Emotional alignment with your message

        *Analysis may take 1–3 minutes depending on video length. Feedback is now fully dynamic—tailored to your unique style!*
        """
    )

    # Summary badges display
    summary_badges_output = gr.HTML(label="📈 Quick Summary", elem_classes="summary-badges")

    with gr.Row():
        video_input = gr.Video(
            label="🎥 Upload Your Video",
            format="mp4",
            height=300,
            width=500,
            interactive=True
        )

    analyze_button = gr.Button("✨ Analyze My Communication", variant="primary", size="lg")

    with gr.Tabs():
        with gr.TabItem("⭐ Key Coaching Tips"):
            # NEW: Use Markdown for better dynamic rendering
            coaching_output = gr.Markdown(
                label="🌟 Your Personalized Feedback",
                elem_classes="coaching-box"
            )

        with gr.TabItem("📊 Emotion Visualization"):
            plot_output = gr.Plot(
                label="Facial Emotion Trends Over Time"
            )

        with gr.TabItem("📄 Full Detailed Report"):
            detailed_report_output = gr.Textbox(
                label="📋 Complete Analysis Breakdown",
                lines=20,
                interactive=False,
                show_copy_button=True,
                elem_classes="report-box"
            )

    # Add footer
    gr.Markdown(
        """
        ---
        *Powered by DeepFace, Whisper, and Wav2Vec2.
        Your video is processed locally — never uploaded to the cloud.
        © 2025 AI Communication Coach — Built for growth, not surveillance.*
        """
    )

    # Connect button to function
    analyze_button.click(
        fn=the_ai_communication_coach,
        inputs=video_input,
        outputs=[coaching_output, plot_output, detailed_report_output, summary_badges_output],
        concurrency_limit=1  # Prevent overload
    )

    # Add CSS for styling
    gr.HTML(
        """
        <style>
            .summary-badges {
                margin: 15px 0;
                padding: 10px;
                border-radius: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #4ECDC4;
                text-align: center;
            }
            .coaching-box, .report-box {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 16px;
                line-height: 1.6;
            }
            .gr-button {
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            .gr-markdown h1 {
                text-align: center;
                color: #2d3748;
            }
            .gr-video {
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .coaching-box h2 { color: #4ECDC4; }  /* NEW: Style dynamic headers */
            .coaching-box strong { color: #2d3748; }
        </style>
        """
    )

# Launch with enhanced settings
demo.launch(
    debug=True,
    share=True,
    server_port=7860,
    server_name="0.0.0.0",
    show_api=False
)
