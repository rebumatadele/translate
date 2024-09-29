import streamlit as st
import openai
from anthropic import Anthropic
import google.generativeai as genai
import time
from dotenv import load_dotenv
import os
from curl_cffi import requests
from nltk import sent_tokenize
import re

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Utility function for error handling
def handle_error(message):
    st.error(message)
    # Optionally, log to a file
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Load prompt from a file
def load_prompt(file_path="prompt.txt"):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        handle_error(f"File {file_path} not found.")
        return ""

# Save the edited prompt back to the file
def save_prompt(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

# Function to configure OpenAI
def configure_openai(api_key):
    openai.api_key = api_key
    
# Sanitize the file name by removing invalid characters
def sanitize_file_name(file_name):
    # Replace invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*\r\n]+', '_', file_name)

# Function to generate response with OpenAI
def generate_with_openai(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,  
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message['content']

# Function to configure Anthropic (Claude)
def configure_anthropic(api_key):
    global ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = api_key

def generate_with_anthropic(prompt):
    headers = {
        'x-api-key': ANTHROPIC_API_KEY,
        'content-type': 'application/json',
        'anthropic-version': '2023-06-01',
    }

    data = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }
    
    try:
        response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            response_json = response.json()
            if "content" in response_json:
                if isinstance(response_json["content"], list):
                    return "".join([item.get("text", "") for item in response_json["content"]])
                elif isinstance(response_json["content"], str):
                    return response_json["content"]
            else:
                return "No content field in response."
        else:
            handle_error(f"Anthropic Error: {response.status_code} - {response.json().get('error', {}).get('message', 'Unknown error')}")
            return None
    except Exception as e:
        handle_error(f"An error occurred: {e}")
        return None

# Function to configure Gemini (Google Generative AI)
def configure_gemini(api_key):
    genai.configure(api_key=api_key)

# Function to generate response with Gemini
def generate_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Function to get the response based on the provider
def get_response(prompt, provider_choice, model_choice=None):
    if provider_choice == "OpenAI":
        return generate_with_openai(prompt, model=model_choice)
    elif provider_choice == "Anthropic":
        return generate_with_anthropic(prompt)
    elif provider_choice == "Gemini":
        return generate_with_gemini(prompt)

# Split text into chunks of words, sentences, or paragraphs
def split_text_into_chunks(text, chunk_size, chunk_by="words"):
    if chunk_by == "words":
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    elif chunk_by == "sentences":
        sentences = sent_tokenize(text)
        return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    elif chunk_by == "paragraphs":
        paragraphs = text.split('\n\n')
        return paragraphs  # Paragraphs are already naturally split

# Load text from a file
def load_text(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        handle_error(f"File {file_path} not found.")
        return ""

# Write response to a file, now with the same name as the original file plus '_done'
def write_response_to_file(response_text, original_file_name):
    # Set default file name if original_file_name is None
    if original_file_name is None:
        original_file_name = "output.txt"  # Default name
        output_file_name = "output_done.txt"
    else:
        file_name_without_ext, ext = os.path.splitext(original_file_name)
        sanitized_name = sanitize_file_name(file_name_without_ext)
        output_file_name = f"{sanitized_name}_done{ext}"
    
    # Write response to file
    with open(output_file_name, 'w') as file:
        file.write(response_text)
    
    return output_file_name

def process_text(text, provider_choice, prompt, chunk_size, chunk_by="words", model_choice=None, progress_bar=None, original_file_name=None):
    final_response = ""
    
    if provider_choice in ["OpenAI", "Anthropic"]:
        chunks = split_text_into_chunks(text, chunk_size, chunk_by)

        for i, chunk in enumerate(chunks):
            combined_prompt = prompt + chunk 
            response = get_response(combined_prompt, provider_choice, model_choice=model_choice)
            if response is not None:
                final_response += response
            else:
                handle_error(f"Failed to get response for chunk {i + 1}.")
            
            if progress_bar:
                progress_bar.progress((i + 1) / len(chunks))
            time.sleep(1)
    else:
        combined_prompt = text + prompt
        response = get_response(combined_prompt, provider_choice)
        if response is not None:
            final_response = response
        else:
            handle_error("Failed to get response for the text.")

    # Check if the final_response is empty
    if final_response.strip():
        output_file_name = write_response_to_file(final_response, original_file_name)
        return final_response, output_file_name
    else:
        handle_error("No valid response was generated. Please check the provider and prompt settings.")
        return None, None

# Streamlit app UI
st.set_page_config(page_title="Text Processor with Generative AI", page_icon="🤖", layout="wide")

# Sidebar for configuration
st.sidebar.title("Configuration")
st.sidebar.subheader("Provider Settings")

# Select provider
provider_choice = st.sidebar.selectbox(
    "Choose a provider",
    ["OpenAI", "Anthropic", "Gemini"]
)

# Add model selection for all providers
model_choice = None
if provider_choice == "OpenAI":
    model_choice = st.sidebar.selectbox("Choose a model", ["gpt-3.5-turbo", "gpt-4"])
elif provider_choice == "Anthropic":
    model_choice = st.sidebar.selectbox("Choose a model", ["claude-3-5-sonnet-20240620", "claude-3-5"])
elif provider_choice == "Gemini":
    model_choice = st.sidebar.selectbox("Choose a model", ["gemini-1.5-flash", "gemini-1.5"])

# Prefill the API key field from environment variables or manual entry
api_key = st.sidebar.text_input("API Key", type="password", value={
    "OpenAI": OPENAI_API_KEY,
    "Anthropic": ANTHROPIC_API_KEY,
    "Gemini": GEMINI_API_KEY
}.get(provider_choice, ""))

# Configure the provider
if st.sidebar.button("Configure"):
    if api_key:
        if provider_choice == "OpenAI":
            configure_openai(api_key)
        elif provider_choice == "Anthropic":
            configure_anthropic(api_key)
        elif provider_choice == "Gemini":
            configure_gemini(api_key)
        st.sidebar.success(f"{provider_choice} configured successfully!")
    else:
        handle_error("Please enter an API key.")

# Main layout
st.title("Text Processor with Generative AI 🤖")
st.subheader("Upload your text files and process them with AI")

# Editable prompt area
st.header("Prompt Template")
prompt_text = load_prompt()
edited_prompt = st.text_area("Edit your prompt template", value=prompt_text, height=200)

# Button to save the edited prompt
if st.button("Save Prompt"):
    save_prompt("prompt.txt", edited_prompt)
    st.success("Prompt saved successfully!")

# Initialize session state for uploaded files and results
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'file_contents' not in st.session_state:
    st.session_state.file_contents = {}

# File uploader for multiple files with a dynamic key
st.header("Upload Files")
uploader_key = "file_uploader_" + str(st.session_state.get("uploader_key", 0))
uploaded_files = st.file_uploader("Upload input text files", type="txt", accept_multiple_files=True, key=uploader_key)

# Store uploaded files in session state
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state.file_contents:
            file_content = uploaded_file.read().decode('utf-8')
            st.session_state.file_contents[file_name] = file_content

# Clear button to clear the uploaded files
if st.session_state.file_contents:
    # st.markdown("<p style='font-size: small;'>Double Tap to clear</p>", unsafe_allow_html=True)
    if st.button("Clear Files and Outputs"):
        st.session_state.uploaded_files = None
        st.session_state.results = []
        st.session_state.file_contents = {}
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
        st.rerun()

# Preview and edit section
if st.session_state.file_contents:
    st.header("Preview and Edit Files")
    selected_file = st.selectbox("Select a file to preview and edit", list(st.session_state.file_contents.keys()), key="file_selector")
    
    # Initialize the edited content in session state if it doesn't exist
    if f"edited_{selected_file}" not in st.session_state:
        st.session_state[f"edited_{selected_file}"] = st.session_state.file_contents[selected_file]
    
    # Use a unique key for the text_area to ensure it updates correctly
    edited_content = st.text_area("Edit the content", value=st.session_state[f"edited_{selected_file}"], height=300, key=f"edit_{selected_file}")
    
    # Update the session state whenever the content changes
    if edited_content != st.session_state[f"edited_{selected_file}"]:
        st.session_state[f"edited_{selected_file}"] = edited_content
    
    if st.button("Save Changes", key=f"save_{selected_file}"):
        st.session_state.file_contents[selected_file] = edited_content
        st.success(f"Changes to {selected_file} saved successfully!")
        st.rerun()
        
# Input for chunk size and chunk type (words, sentences, paragraphs)
st.header("Processing Settings")
chunk_size_input = st.number_input("Set chunk size", min_value=1, max_value=5000, value=500)
chunk_by = st.selectbox("Chunk by", ["words", "sentences", "paragraphs"])


# Button to process the files
if st.button("Process Text"):
    if st.session_state.file_contents:
        results = []
        progress_bar = st.progress(0)
        for i, (file_name, file_content) in enumerate(st.session_state.file_contents.items()):
            response_text, output_file_name = process_text(file_content, provider_choice, edited_prompt, chunk_size_input, chunk_by, model_choice=model_choice, progress_bar=progress_bar, original_file_name=file_name)
            
            results.append((response_text, output_file_name))
            
            progress_bar.progress((i + 1) / len(st.session_state.file_contents))
        
        st.session_state.results = results


# Display results
st.header("Results")
for response_text, output_file_name in st.session_state.results:
    st.subheader(f"Output for {output_file_name}")
    st.text_area(f"Output for {output_file_name}", value=response_text, height=300)
    st.download_button(
        label=f"Download Output for {output_file_name}",
        data=response_text,
        file_name=output_file_name,
        mime="text/plain"
    )