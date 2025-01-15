import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import pipeline, AutoTokenizer

# Initialize summarization pipeline and tokenizer
try:
    text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
except Exception as e:
    st.error(f"Error initializing summarization pipeline: {e}")

def split_text_into_chunks(text, max_tokens=1024):
    """Splits the text into chunks that fit within the model's token limit."""
    words = text.split()
    current_chunk = []
    current_length = 0
    chunks = []

    for word in words:
        token_count = len(tokenizer.tokenize(word))
        if current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summary(input_text):
    """Summarizes the input text, handling token limits."""
    summaries = []
    chunks = split_text_into_chunks(input_text)
    for chunk in chunks:
        try:
            output = text_summary(chunk)
            summaries.append(output[0]['summary_text'])
        except Exception as e:
            return f"Error summarizing text: {e}"
    return " ".join(summaries)

def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_transcript(video_url):
    """Fetches the transcript of a YouTube video and summarizes it."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Video ID could not be extracted."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript:
            return "No transcript available for this video."

        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        summary_text = summary(text_transcript)
        return summary_text
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit App Layout
st.title("@GenAILearniverse Project 2: YouTube Script Summarizer")
st.write("This application summarizes the YouTube video script.")

# Input field for YouTube URL
video_url = st.text_input("Input YouTube URL to summarize")

# Display summary when user provides a URL
if st.button("Summarize"):
    if video_url.strip():
        with st.spinner("Processing..."):
            summary_result = get_youtube_transcript(video_url)
            st.text_area("Summarized text", summary_result, height=300)
    else:
        st.warning("Please enter a valid YouTube URL.")
