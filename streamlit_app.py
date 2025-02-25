import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
import os
import wave
from collections import Counter
import langid
from wordcloud import WordCloud
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, ClientSettings
import av

# Set up Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to send the audio file to the API
def transcribe_audio(file):
    try:
        data = file.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        if response.status_code == 200:
            return response.json()  # Return transcription
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Function for keyword extraction using CountVectorizer
def extract_keywords(text):
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

# Function to detect language of the text
def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang, confidence

# Function to calculate speech rate (words per minute)
def calculate_speech_rate(text, duration_seconds):
    words = text.split()
    num_words = len(words)
    if duration_seconds > 0:
        speech_rate = num_words / (duration_seconds / 60)
    else:
        speech_rate = 0
    return speech_rate

# Function to generate a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Custom audio processor to save audio frames
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []
    
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame)
        return frame

# Streamlit UI
st.title("🎙️ Voice to Story Creator")
st.write("Upload or record an audio file, and this app will transcribe it using OpenAI Whisper via Hugging Face API. It will then perform various analyses and generate a creative story or novel.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

# Audio recording
st.subheader("Or record your audio")
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True},
    ),
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor
    if st.button("Stop Recording"):
        audio_frames = audio_processor.frames
        with wave.open("recorded_audio.wav", "wb") as f:
            # Set parameters for the wave file (e.g., mono channel, 16-bit depth, 44.1 kHz)
            f.setnchannels(1)
            f.setsampwidth(2)  # 16-bit depth
            f.setframerate(44100)
            for frame in audio_frames:
                f.writeframes(frame.to_ndarray().tobytes())
        uploaded_file = open("recorded_audio.wav", "rb")

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/mp3", start_time=0)
    st.info("Transcribing audio... Please wait.")
    
    # Transcribe the uploaded audio file
    result = transcribe_audio(uploaded_file)
    
    # Display the result
    if "text" in result:
        st.success("Transcription Complete:")
        transcription_text = result["text"]
        st.write(transcription_text)

        # Language Detection
        lang, confidence = detect_language(transcription_text)
        st.subheader("Language Detection")
        st.write(f"Detected Language: {lang}, Confidence: {confidence}")

        # Keyword Extraction
        keywords = extract_keywords(transcription_text)
        st.subheader("Keyword Extraction")
        st.write(keywords)

        # Speech Rate Calculation
        try:
            duration_seconds = len(uploaded_file.read()) / (44100 * 2)  # Assuming 44.1kHz sample rate and 16-bit samples
            speech_rate = calculate_speech_rate(transcription_text, duration_seconds)
            st.subheader("Speech Rate")
            st.write(f"Speech Rate: {speech_rate} words per minute")
        except ZeroDivisionError:
            st.error("Error: The duration of the audio is zero, which caused a division by zero error.")

        # Word Cloud Visualization
        wordcloud = generate_word_cloud(transcription_text)
        st.subheader("Word Cloud")
        st.image(wordcloud.to_array())

        # Add download button for the transcription text
        st.download_button(
            label="Download Transcription",
            data=transcription_text,
            file_name="transcription.txt",
            mime="text/plain"
        )

        # Add download button for analysis results
        analysis_results = f"""
        Language Detection:
        Detected Language: {lang}, Confidence: {confidence}
        
        Keyword Extraction:
        {keywords}
        
        Speech Rate: {speech_rate} words per minute
        """
        st.download_button(
            label="Download Analysis Results",
            data=analysis_results,
            file_name="analysis_results.txt",
            mime="text/plain"
        )

        # Generative AI Analysis
        st.subheader("Generative AI Analysis")
        prompt = f"Create a creative story based on the following transcription: {transcription_text}"
        
        # Let user decide if they want to use AI to generate a story
        if st.button("Generate Story"):
            try:
                # Load and configure the model
                model = genai.GenerativeModel(model_name="text-davinci-003")  # Use OpenAI's GPT model
                # Generate response from the model
                response = model.generate_content(prompt)
                
                # Display response in Streamlit
                st.write("Generated Story:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
