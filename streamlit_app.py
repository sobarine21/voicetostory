import streamlit as st
import requests
import wave
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import langid
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import google.generativeai as genai

# Set up Hugging Face API details (for transcription)
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Configure the generative AI API with your Google API key from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Function to send the audio file to the API for transcription
def transcribe_audio(file):
    try:
        data = file.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        if response.status_code == 200:
            result = response.json()
            if "text" in result:
                return result  # Return transcription text
            else:
                return {"error": f"API response did not contain 'text': {response.text}"}
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

# Function to calculate duration of audio file in seconds
def get_audio_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        return 0  # If error occurs, assume zero duration

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

# Function to generate a word cloud from text
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

# File uploader for audio files
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

# Audio recording section
st.subheader("Or record your audio")
rtc_configuration = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"audio": True},
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor
    if st.button("Stop Recording"):
        audio_frames = audio_processor.frames
        # Save the recorded audio frames as a WAV file
        with wave.open("recorded_audio.wav", "wb") as f:
            f.setnchannels(1)  # Mono channel
            f.setsampwidth(2)  # 16-bit depth
            f.setframerate(44100)  # 44.1 kHz sample rate
            for frame in audio_frames:
                f.writeframes(frame.to_ndarray().tobytes())
        
        uploaded_file = open("recorded_audio.wav", "rb")

# Handle uploaded audio file and transcription
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3", start_time=0)
    st.info("Transcribing audio... Please wait.")
    
    # Transcribe the uploaded audio file
    result = transcribe_audio(uploaded_file)
    
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
        file_path = "recorded_audio.wav"
        duration_seconds = get_audio_duration(file_path)
        speech_rate = calculate_speech_rate(transcription_text, duration_seconds)
        st.subheader("Speech Rate")
        st.write(f"Speech Rate: {speech_rate} words per minute")

        # Word Cloud Visualization
        wordcloud = generate_word_cloud(transcription_text)
        st.subheader("Word Cloud")
        st.image(wordcloud.to_array())

        # Add download button for the transcription
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
                # Load and configure the model with Google's `gemini-1.5-flash`
                model = genai.GenerativeModel('gemini-1.5-flash')
                
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
