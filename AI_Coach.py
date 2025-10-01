import gradio as gr
import numpy as np
import re
import time
import cv2
import os
import tempfile
from transformers import pipeline
import whisper
import subprocess
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load pre-trained models
sentiment_analyzer = pipeline("sentiment-analysis")
whisper_model = whisper.load_model("base")  # Choose size: tiny, base, small

# Define filler words to track
FILLER_WORDS = ["um", "uh", "like", "you know", "actually", "basically", "literally", 
                "sort of", "kind of", "so", "well", "just", "stuff", "things"]

# Define emotions for tracking and visualization
EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral"]

# Extract audio from video
def extract_audio(video_path):
    audio_path = tempfile.mktemp(suffix='.mp3')
    
    try:
        # Use ffmpeg to extract audio from video
        command = [
            "ffmpeg", 
            "-i", video_path, 
            "-q:a", "0", 
            "-map", "a", 
            audio_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

# Simple emotion estimation using face features
def estimate_emotion(face_img):
    # This is a simplified placeholder for emotion detection
    # In a real implementation, you would use a proper emotion classification model
    
    if face_img is None or face_img.size == 0:
        return {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, "surprise": 0.0}
    
    # Resize for consistent processing
    face_img = cv2.resize(face_img, (48, 48))
    
    # We'll use simple heuristics based on pixel intensities and gradients
    # Extract different facial regions
    h, w = face_img.shape
    
    # Divide face into regions (approximate locations)
    forehead = face_img[0:int(h*0.3), :]
    eyes = face_img[int(h*0.2):int(h*0.5), :]
    mouth = face_img[int(h*0.6):h, :]
    
    # Calculate features
    avg_intensity = np.mean(face_img)
    mouth_variance = np.var(mouth)
    eye_variance = np.var(eyes)
    
    # Horizontal gradient in mouth area (for smile detection)
    # High gradient can indicate a smile
    mouth_gradient_x = cv2.Sobel(mouth, cv2.CV_64F, 1, 0, ksize=3)
    mouth_gradient_y = cv2.Sobel(mouth, cv2.CV_64F, 0, 1, ksize=3)
    mouth_gradient_mag = np.sqrt(mouth_gradient_x**2 + mouth_gradient_y**2)
    mouth_gradient = np.mean(mouth_gradient_mag)
    
    # Eye region gradient (for surprise/anger detection)
    eye_gradient_x = cv2.Sobel(eyes, cv2.CV_64F, 1, 0, ksize=3)
    eye_gradient_y = cv2.Sobel(eyes, cv2.CV_64F, 0, 1, ksize=3)
    eye_gradient_mag = np.sqrt(eye_gradient_x**2 + eye_gradient_y**2)
    eye_gradient = np.mean(eye_gradient_mag)
    
    # Default to neutral
    emotions = {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "surprise": 0.0,
        "neutral": 0.6  # Start with neutral as base
    }
    
    # Happy: Higher mouth gradient, higher mouth variance
    if mouth_gradient > 10 and mouth_variance > 300:
        emotions["happy"] = 0.7
        emotions["neutral"] -= 0.3
    
    # Sad: Lower mouth region intensity relative to eyes
    mouth_eye_ratio = np.mean(mouth) / np.mean(eyes) if np.mean(eyes) > 0 else 1
    if mouth_eye_ratio < 0.8:
        emotions["sad"] = 0.5
        emotions["neutral"] -= 0.2
    
    # Angry: High eye gradient, low mouth variance
    if eye_gradient > 15 and mouth_variance < 200:
        emotions["angry"] = 0.6
        emotions["neutral"] -= 0.2
    
    # Surprise: High eye variance, high overall variance
    if eye_variance > 500 and np.var(face_img) > 800:
        emotions["surprise"] = 0.6
        emotions["neutral"] -= 0.2
    
    # Ensure no negative values
    for emotion in emotions:
        emotions[emotion] = max(0.0, emotions[emotion])
    
    # Normalize to sum to 1
    total = sum(emotions.values())
    if total > 0:
        for emotion in emotions:
            emotions[emotion] /= total
    
    return emotions

# Face detection function using OpenCV
def detect_faces(frame):
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame  # Already grayscale
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces, gray

# Create emotion radar chart
def create_emotion_radar_chart(emotions):
    # Data preparation
    emotions_list = list(emotions.items())
    emotions_list.sort(key=lambda x: x[1], reverse=True)  # Sort by value
    labels = [emotion[0].capitalize() for emotion in emotions_list]
    values = [emotion[1] for emotion in emotions_list]
    
    # Create figure and polar subplot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(labels)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add values (and close the loop)
    values += values[:1]
    
    # Draw the chart
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    plt.xticks(angles[:-1], labels)
    
    # Set y limits
    ax.set_ylim(0, 1)
    
    # Add title
    plt.title('Emotion Distribution', size=15, color='navy', y=1.1)
    
    # Save to BytesIO and encode as base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Encode the image to base64
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{data}"

def analyze_video(video_file):
    if video_file is None:
        return "Please upload a video file for analysis.", "", None
    
    result_dict = {"audio_analysis": {}, "visual_analysis": {}}
    start_time = time.time()
    
    # === EXTRACT AUDIO FROM VIDEO ===
    audio_file = extract_audio(video_file)
    if not audio_file:
        return "Failed to extract audio from video. Please check the format and try again.", "", None
    
    # === AUDIO ANALYSIS ===
    # Transcribe audio using Whisper
    audio_result = whisper_model.transcribe(audio_file)
    transcription = audio_result["text"]
    
    if not transcription.strip():
        return "No speech detected in the video. Please try again with clear audio.", "", None
    
    # Perform sentiment analysis
    sentiment_result = sentiment_analyzer(transcription)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']
    
    # Count words and calculate speech metrics
    words = transcription.split()
    total_words = len(words)
    
    # Calculate speech duration in minutes from the audio file
    try:
        import librosa
        audio_duration = librosa.get_duration(filename=audio_file)
    except:
        audio_duration = audio_result.get("duration", 0)
    
    duration_minutes = audio_duration / 60
    
    # Calculate words per minute (WPM)
    wpm = int(total_words / duration_minutes) if duration_minutes > 0 else 0
    
    # Count filler words
    filler_count = 0
    filler_instances = []
    
    lower_transcription = transcription.lower()
    for filler in FILLER_WORDS:
        matches = re.finditer(r'\b' + re.escape(filler) + r'\b', lower_transcription)
        for match in matches:
            filler_count += 1
            filler_instances.append(filler)
    
    filler_percentage = (filler_count / total_words) * 100 if total_words > 0 else 0
    
    # Store audio analysis results
    result_dict["audio_analysis"] = {
        "transcription": transcription,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "total_words": total_words,
        "duration_minutes": duration_minutes,
        "wpm": wpm,
        "filler_count": filler_count,
        "filler_percentage": filler_percentage,
        "filler_instances": filler_instances
    }
    
    # === VISUAL ANALYSIS ===
    visual_analysis = {
        "face_detections": 0, 
        "frame_count": 0, 
        "eye_contact": 0,
        "emotions": {emotion: 0 for emotion in EMOTIONS}
    }
    sample_frames = []
    emotion_frame_counts = {emotion: 0 for emotion in EMOTIONS}
    
    # Process video for face detection and emotion analysis
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Initialize counters
    face_detected_frames = 0
    center_face_frames = 0
    face_sizes = []
    all_emotions = {emotion: [] for emotion in EMOTIONS}
    
    # Sample frames (analyze every 15th frame to save processing time)
    sample_rate = 15
    frame_idx = 0
    
    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as temp_dir:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % sample_rate != 0:  # Skip frames for efficiency
                continue
            
            visual_analysis["frame_count"] += 1
            
            try:
                # Detect faces
                faces, gray = detect_faces(frame)
                
                if len(faces) > 0:
                    face_detected_frames += 1
                    
                    # Get the largest face (presumably the speaker)
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Save face size relative to frame
                    face_size_percent = (w * h) / (frame.shape[1] * frame.shape[0]) * 100
                    face_sizes.append(face_size_percent)
                    
                    # Extract face for emotion analysis
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Analyze emotion with our simple classifier
                    emotion_scores = estimate_emotion(face_roi)
                    
                    # Get dominant emotion
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                    
                    # Track emotions
                    for emotion in EMOTIONS:
                        score = emotion_scores.get(emotion, 0)
                        all_emotions[emotion].append(score)
                    
                    # Count frames with this dominant emotion
                    emotion_frame_counts[dominant_emotion[0]] += 1
                    
                    # Draw rectangle and emotion on frame
                    frame_with_analysis = frame.copy()
                    cv2.rectangle(frame_with_analysis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add emotion label
                    emotion_text = f"{dominant_emotion[0]}: {dominant_emotion[1]:.0%}"
                    cv2.putText(frame_with_analysis, emotion_text, 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check if face is centered in frame (simple eye contact proxy)
                    frame_center_x = frame.shape[1] / 2
                    frame_center_y = frame.shape[0] / 2
                    face_center_x = x + w/2
                    face_center_y = y + h/2
                    
                    # If face is reasonably close to center, count as eye contact
                    if (abs(face_center_x - frame_center_x) < frame.shape[1] * 0.2 and
                        abs(face_center_y - frame_center_y) < frame.shape[0] * 0.2):
                        center_face_frames += 1
                    
                    # Save sample frames with different dominant emotions
                    # Try to get samples of different emotions
                    if len(sample_frames) < 5:
                        # Check if we already have a frame with this emotion
                        emotion_exists = False
                        for _, frame_emotion in sample_frames:
                            if frame_emotion == dominant_emotion[0]:
                                emotion_exists = True
                                break
                                
                        # Save if new emotion or we need more samples
                        if not emotion_exists or len(sample_frames) < 3:
                            frame_path = os.path.join(temp_dir, f"{dominant_emotion[0]}_{len(sample_frames)}.jpg")
                            cv2.imwrite(frame_path, frame_with_analysis)
                            sample_frames.append((frame_path, dominant_emotion[0]))
            
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        
        cap.release()
        
        # Calculate percentages and averages
        face_detected_percentage = (face_detected_frames / visual_analysis["frame_count"]) * 100 if visual_analysis["frame_count"] > 0 else 0
        eye_contact_percentage = (center_face_frames / face_detected_frames) * 100 if face_detected_frames > 0 else 0
        
        # Calculate average face size (proxy for distance from camera)
        avg_face_size = sum(face_sizes) / len(face_sizes) if face_sizes else 0
        
        # Calculate average emotion scores
        for emotion in EMOTIONS:
            if all_emotions[emotion]:
                visual_analysis["emotions"][emotion] = sum(all_emotions[emotion]) / len(all_emotions[emotion])
        
        # Store visual analysis results
        visual_analysis["face_detections"] = face_detected_percentage
        visual_analysis["eye_contact"] = eye_contact_percentage
        visual_analysis["avg_face_size"] = avg_face_size
        visual_analysis["duration"] = duration
        visual_analysis["emotion_frame_counts"] = emotion_frame_counts
        
        # Create emotion radar chart
        radar_chart = create_emotion_radar_chart(visual_analysis["emotions"])
        
        # Create a composite image of sample frames if available
        sample_image = None
        if sample_frames:
            # Create a composite image showing different sample frames
            sample_imgs = []
            for img_path, emotion in sample_frames:
                img = cv2.imread(img_path)
                sample_imgs.append(img)
            
            # If we have samples, create a horizontal stack
            if sample_imgs:
                # Resize images to same height
                height = min(sample_imgs[0].shape[0], 200)
                resized_imgs = []
                for img, (_, emotion) in zip(sample_imgs, sample_frames):
                    aspect = img.shape[1] / img.shape[0]
                    width = int(height * aspect)
                    resized = cv2.resize(img, (width, height))
                    resized_imgs.append(resized)
                
                # Create a combined image
                sample_image = np.hstack(resized_imgs)
    
    result_dict["visual_analysis"] = visual_analysis
    
    # === GENERATE REPORT ===
    # Audio analysis suggestions
    audio_suggestions = []
    
    # Suggestion for speech rate
    if wpm < 120:
        audio_suggestions.append("Your speaking pace is slow. Try to speak a bit faster to maintain audience engagement.")
    elif wpm > 160:
        audio_suggestions.append("Your speaking pace is quite fast. Consider slowing down to improve clarity.")
    else:
        audio_suggestions.append("Your speaking pace is good, within the ideal range of 120-160 words per minute.")
    
    # Suggestion for filler words
    if filler_percentage > 5:
        audio_suggestions.append(f"You used filler words frequently ({filler_percentage:.1f}% of words). Try to reduce the use of: {', '.join(set(filler_instances))}.")
    elif filler_percentage > 0:
        audio_suggestions.append(f"You used some filler words ({filler_percentage:.1f}% of words). Be mindful of: {', '.join(set(filler_instances))}.")
    else:
        audio_suggestions.append("Great job avoiding filler words!")
    
    # Visual analysis suggestions
    visual_suggestions = []
    
    # Visibility
    if visual_analysis["face_detections"] < 70:
        visual_suggestions.append(f"You were visible in the frame only {visual_analysis['face_detections']:.0f}% of the time. Try to stay in the camera view.")
    else:
        visual_suggestions.append(f"Good camera presence! You were visible {visual_analysis['face_detections']:.0f}% of the time.")
        
    # Eye contact
    if visual_analysis["eye_contact"] < 50:
        visual_suggestions.append(f"You maintained eye contact approximately {visual_analysis['eye_contact']:.0f}% of the time. Try to look at the camera more consistently.")
    elif visual_analysis["eye_contact"] >= 70:
        visual_suggestions.append(f"Excellent eye contact maintained ({visual_analysis['eye_contact']:.0f}% of the time).")
    else:
        visual_suggestions.append(f"You maintained good eye contact ({visual_analysis['eye_contact']:.0f}% of the time). Continue to engage with your audience.")
    
    # Distance from camera
    if visual_analysis["avg_face_size"] < 5:
        visual_suggestions.append("You appear to be too far from the camera. Consider moving closer for better engagement.")
    elif visual_analysis["avg_face_size"] > 25:
        visual_suggestions.append("You appear to be very close to the camera. Consider moving back slightly for a more professional framing.")
    
    # Emotional expression
    dominant_emotion = max(visual_analysis["emotions"].items(), key=lambda x: x[1])
    if dominant_emotion[1] > 0.5:
        if dominant_emotion[0] == "neutral":
            visual_suggestions.append(f"Your expressions are predominantly neutral ({dominant_emotion[1]:.0%}). Consider incorporating more varied expressions to engage your audience.")
        elif dominant_emotion[0] == "happy":
            visual_suggestions.append(f"Your expressions are mostly positive ({dominant_emotion[1]:.0%}), which helps establish rapport with your audience.")
        elif dominant_emotion[0] in ["sad", "angry"]:
            visual_suggestions.append(f"Your expressions show significant {dominant_emotion[0]} emotion ({dominant_emotion[1]:.0%}). This may affect how your message is received.")
    
    # Evaluate emotional variety
    num_significant_emotions = sum(1 for emotion, score in visual_analysis["emotions"].items() if score > 0.2)
    if num_significant_emotions <= 1:
        visual_suggestions.append("Your facial expressions lack variety. Try to be more expressive to keep your audience engaged.")
    elif num_significant_emotions >= 3:
        visual_suggestions.append("You display a good range of emotions, which makes your presentation dynamic and engaging.")
    
    # Determine sentiment category
    if sentiment_label == "POSITIVE":
        emoji = "😊"
        color = "#28a745"
    elif sentiment_label == "NEGATIVE":
        emoji = "😞"
        color = "#dc3545"
    else:
        emoji = "😐"
        color = "#ffc107"
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Format the output HTML
    output = f"<div style='margin: 20px;'>"
    
    # 1. Summary section
    output += f"<h1>Speech and Presentation Analysis</h1>"
    output += f"<div style='background-color: #f0f8ff; padding: 15px; border-radius: 5px; color:black; margin-bottom: 20px;'>"
    output += f"<h2>Executive Summary</h2>"
    output += f"<p><b>Speech Sentiment:</b> {sentiment_label} {emoji} ({sentiment_score:.0%} confidence)</p>"
    output += f"<p><b>Speaking Pace:</b> {wpm} words per minute</p>"
    output += f"<p><b>Filler Words:</b> {filler_percentage:.1f}% of speech</p>"
    output += f"<p><b>Camera Presence:</b> {visual_analysis['face_detections']:.0f}% of video</p>"
    output += f"<p><b>Eye Contact:</b> {visual_analysis['eye_contact']:.0f}% of visible time</p>"
    output += f"<p><b>Dominant Expression:</b> {dominant_emotion[0]} ({dominant_emotion[1]:.0%})</p>"
    output += "</div>"
    
    # 2. Transcription
    output += f"<h2>Transcription:</h2>"
    output += f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; color:black; margin-bottom: 20px;'>{transcription}</div>"
    
    # 3. Audio Analysis
    output += f"<h2>Speech Analysis:</h2>"
    output += f"<div style='margin-bottom: 20px;'>"
    output += f"<div style='text-align: center; margin: 15px; padding: 10px; border-radius: 5px; background-color: {color}20;'>"
    output += f"<h3 style='color: {color};'>{sentiment_label} {emoji}</h3>"
    output += f"<p>Confidence: {sentiment_score:.0%}</p>"
    output += "</div>"
    
    output += f"<ul>"
    output += f"<li><b>Total Words:</b> {total_words}</li>"
    output += f"<li><b>Speech Duration:</b> {duration_minutes:.2f} minutes</li>"
    output += f"<li><b>Speaking Rate:</b> {wpm} words per minute</li>"
    output += f"<li><b>Filler Words:</b> {filler_count} ({filler_percentage:.1f}% of total words)</li>"
    if filler_count > 0:
        output += f"<li><b>Common Fillers:</b> {', '.join([f'{word} ({filler_instances.count(word)})' for word in set(filler_instances)])}</li>"
    output += f"</ul>"
    output += "</div>"
    
    # 4. Visual Analysis with improved emotion display
    output += f"<h2>Visual Analysis:</h2>"
    output += f"<div style='margin-bottom: 20px;'>"
    
    # Add the radar chart
    output += f"<div style='text-align: center; margin-bottom: 20px;'>"
    output += f"<h3>Emotion Distribution:</h3>"
    output += f"<img src='{radar_chart}' alt='Emotion Radar Chart' style='max-width: 100%; height: auto;'>"
    output += "</div>"
    
    # Emotion percentages as bar chart
    output += f"<h3>Emotional Expression Details:</h3>"
    output += f"<div style='margin-bottom: 15px;'>"
    for emotion, percentage in sorted(visual_analysis["emotions"].items(), key=lambda x: x[1], reverse=True):
        # Map emotions to colors
        if emotion == "happy":
            bar_color = "#28a745"  # green
        elif emotion == "sad":
            bar_color = "#6c757d"  # gray
        elif emotion == "angry":
            bar_color = "#dc3545"  # red
        elif emotion == "surprise":
            bar_color = "#ffc107"  # yellow
        else:
            bar_color = "#17a2b8"  # teal (neutral)
            
        # Create bar chart
        bar_width = int(percentage * 100)
        output += f"<div style='margin-bottom: 8px;'>"
        output += f"<div style='display: flex; align-items: center;'>"
        output += f"<div style='width: 80px;'><b>{emotion.capitalize()}:</b></div>"
        output += f"<div style='flex-grow: 1; background-color: #e9ecef; border-radius: 4px; height: 20px;'>"
        output += f"<div style='background-color: {bar_color}; height: 20px; width: {bar_width}%; min-width: 10px; border-radius: 4px; text-align: center; color: white;'>"
        output += f"{percentage:.0%}" if bar_width > 15 else ""
        output += f"</div></div>"
        output += f"<div style='margin-left: 10px; width: 50px;'>{percentage:.0%}</div>"
        output += f"</div></div>"
    output += f"</div>"
    
    # Other visual metrics
    output += f"<h3>Visual Presence Metrics:</h3>"
    output += f"<ul>"
    output += f"<li><b>Face Visibility:</b> {visual_analysis['face_detections']:.0f}% of the video</li>"
    output += f"<li><b>Eye Contact Estimation:</b> {visual_analysis['eye_contact']:.0f}% of visible time</li>"
    output += f"<li><b>Distance from Camera:</b> {'Good' if 5 <= visual_analysis['avg_face_size'] <= 25 else 'Needs adjustment'}</li>"
    output += f"<li><b>Emotional Variety:</b> {num_significant_emotions} distinct expressions</li>"
    output += f"</ul>"
    output += "</div>"
    
    # 5. Improvement Suggestions
    output += f"<h2>Improvement Suggestions:</h2>"
    output += f"<div style='background-color: #f5f5f5; padding: 15px; border-radius: 5px; color:black; margin-bottom: 20px;'>"
    output += f"<h3>Speech Suggestions:</h3>"
    output += f"<ul class='suggestions'>"
    for suggestion in audio_suggestions:
        output += f"<li>{suggestion}</li>"
    output += f"</ul>"
    
    output += f"<h3>Visual Presentation Suggestions:</h3>"
    output += f"<ul class='suggestions'>"
    for suggestion in visual_suggestions:
        output += f"<li>{suggestion}</li>"
    output += f"</ul>"
    output += "</div>"
    
    # 6. Style for better presentation
    output += f"""
    <style>
        ul.suggestions li {{
            margin-bottom: 8px;
            line-height: 1.5;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
    </style>
    """
    
    output += f"<p><i>Analysis completed in {processing_time:.2f} seconds</i></p>"
    output += "</div>"
    
    # Clean up the temporary audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)
    
    return output, transcription, sample_image

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_video,
    inputs=[
        gr.Video(label="Upload or Record Video")
    ],
    outputs=[
        gr.HTML(label="Comprehensive Analysis"),
        gr.Textbox(label="Transcription Text"),
        gr.Image(label="Expression Samples", visible=True)
    ],
    title="Advanced Speech and Visual Presentation Analyzer",
    description="""
    Upload a video to receive a comprehensive analysis of your presentation skills. 
    This tool analyzes:
    • Speech transcription and sentiment
    • Speaking pace and filler words
    • Facial expressions and emotions
    • Eye contact and camera presence
    
    Get actionable feedback to improve your communication skills!
    """,
    examples=[]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
