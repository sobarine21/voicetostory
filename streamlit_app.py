import streamlit as st
import google.generativeai as genai
import requests
from sklearn.feature_extraction.text import CountVectorizer
import wave
import langid
from wordcloud import WordCloud
import io

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
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='coolwarm').generate(text)
    return wordcloud

# Streamlit UI setup with a tech theme
st.set_page_config(page_title="Voice Notes Stories Enhanced by AI", page_icon="🎙️", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #f5f5f5;
        }
        .stButton>button {
            background-color: #6200ea;
            color: white;
            font-weight: bold;
            border-radius: 12px;
        }
        .stButton>button:hover {
            background-color: #3700b3;
        }
        .stTextArea textarea {
            background-color: #333;
            color: #fff;
        }
        .stFileUploader {
            color: #fff;
            background-color: #333;
        }
        h1 {
            color: #6200ea;
        }
        .stProgress {
            background-color: #6200ea;
        }
    </style>
""", unsafe_allow_html=True)

# UI Elements
st.title("🎙️ Voice Notes Stories Enhanced by AI")
st.write("Upload an audio file, and the app will transcribe it using OpenAI Whisper via Hugging Face API. It will also analyze the text and generate a creative story using Generative AI.")

# File uploader with file size limit for 2 minutes (based on file duration and typical bitrates)
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Estimate file duration for 2-minute limit
    try:
        with wave.open(uploaded_file, 'rb') as audio_file:
            duration_seconds = audio_file.getnframes() / audio_file.getframerate()
            if duration_seconds > 120:
                st.error("The audio file exceeds the 2-minute limit. Please upload a shorter file.")
            else:
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
                        speech_rate = calculate_speech_rate(transcription_text, duration_seconds)
                        st.subheader("Speech Rate")
                        st.write(f"Speech Rate: {speech_rate:.2f} words per minute")
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
                    
                    Speech Rate: {speech_rate:.2f} words per minute
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
                            # Load and configure the model with Google's gemini-1.5-flash
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

