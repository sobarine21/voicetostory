import streamlit as st
import google.generativeai as genai
import requests
from sklearn.feature_extraction.text import CountVectorizer
import wave
import langid
from wordcloud import WordCloud
import io
import time

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
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Streamlit UI
st.set_page_config(page_title="🎙️ Voice to Story Creator", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background: #1d1f27;
        color: #ffffff;
    }
    h1 {
        color: #00adb5;
        text-align: center;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #00adb5;
        color: #ffffff;
        border-radius: 8px;
        transition: background-color 0.3s, transform 0.3s;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #007b7f;
        transform: scale(1.05);
    }
    .stFileUploader {
        border: 2px dashed #00adb5;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        padding: 10px;
    }
    .stAlert {
        font-size: 16px;
        background-color: #33363d;
    }
    .stImage {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
    }
    .stMarkdown {
        color: #dddddd;
    }
    .stProgress {
        background-color: #007b7f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Voice to Story Creator")
st.write("Upload an audio file (e.g., .wav, .flac, .mp3), and this app will transcribe it using OpenAI Whisper via Hugging Face API. Then, perform various analyses and generate a creative story.")

# File uploader with file size limit (2 mins of audio)
uploaded_file = st.file_uploader("Upload your audio file (max duration: 2 minutes)", type=["wav", "flac", "mp3"])

# Limit audio to 2 minutes
MAX_DURATION_SECONDS = 120  # 2 minutes

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/mp3", start_time=0)

    # Checking the duration of the audio file
    try:
        audio = uploaded_file.getvalue()
        with wave.open(io.BytesIO(audio), 'rb') as audio_file:
            framerate = audio_file.getframerate()
            frames = audio_file.getnframes()
            duration_seconds = frames / float(framerate)
            
            if duration_seconds > MAX_DURATION_SECONDS:
                st.error(f"Error: Audio duration exceeds the 2-minute limit. Your audio is {duration_seconds:.2f} seconds.")
            else:
                # Add a loading animation while transcription happens
                with st.spinner("Transcribing audio... Please wait."):
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
                            st.write(f"Speech Rate: {speech_rate} words per minute")
                        except ZeroDivisionError:
                            st.error("Error: The duration of the audio is zero, which caused a division by zero error.")

                        # Word Cloud Visualization
                        wordcloud = generate_word_cloud(transcription_text)
                        st.subheader("Word Cloud")
                        st.image(wordcloud.to_array(), use_column_width=True)

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
    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
