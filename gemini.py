import streamlit as st
import openai
from anthropic import Anthropic
import google.generativeai as genai
import time
from dotenv import load_dotenv
import os
import curl_cffi

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Load prompt from a file
def load_prompt(file_path="prompt.txt"):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"File {file_path} not found.")
        return ""

# Save the edited prompt back to the file
def save_prompt(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

# Function to configure OpenAI
def configure_openai(api_key):
    openai.api_key = api_key

# Function to generate response with OpenAI
def generate_with_openai(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",  # Use GPT-4 model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content

# Function to configure Anthropic (Claude)
def configure_anthropic(api_key):
    global ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = api_key

# Function to generate response with Anthropic (Claude)
def generate_with_anthropic(prompt):
    # Check if API key is configured
    if not ANTHROPIC_API_KEY:
        st.error("Anthropic API key is not configured. Please provide a valid API key.")
        return None

    headers = {
        'x-api-key': ANTHROPIC_API_KEY,  # Use x-api-key for authentication
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    data = {
        "model": "claude-3-5-sonnet-20240620",  # Ensure your key has access to this model
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100
    }

    try:
        # Use curl_cffi for the request
        response = curl_cffi.requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data, timeout=10)
        
        # Handle the response
        if response.status_code == 200:
            print("Response:", response.json().get("completion", "No completion field in response"))
            return response.json().get("completion", "No completion field in response")
        elif response.status_code == 403:
            st.error("403 Error: Access forbidden. Please check your API key permissions and ensure it has the correct scope.")
            return None
        else:
            st.error(f"Anthropic Error: {response.status_code} - {response.json().get('error', {}).get('message', 'Unknown error')}")
            return None
    except curl_cffi.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
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
def get_response(prompt, provider_choice):
    if provider_choice == "OpenAI":
        return generate_with_openai(prompt)
    elif provider_choice == "Anthropic":
        return generate_with_anthropic(prompt)
    elif provider_choice == "Gemini":
        return generate_with_gemini(prompt)

# Split text into chunks of words
def split_text_into_chunks(text, chunk_size_in_words):
    words = text.split()
    return [' '.join(words[i:i + chunk_size_in_words]) for i in range(0, len(words), chunk_size_in_words)]

# Load text from a file
def load_text(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"File {file_path} not found.")
        return ""

# Write response to a file
def write_response_to_file(response_text, file_name="output.txt"):
    with open(file_name, 'w') as file:
        file.write(response_text)

# Process the text and save to output file
def process_text(text_or_file, provider_choice, prompt, chunk_size_in_words, progress_bar=None):
    text = load_text(text_or_file)
    if not text:
        return
    
    final_response = ""
    
    if provider_choice in ["OpenAI", "Anthropic"]:
        chunks = split_text_into_chunks(text, chunk_size_in_words)

        for i, chunk in enumerate(chunks):
            combined_prompt = chunk + prompt
            response = get_response(combined_prompt, provider_choice)
            if response is not None:
                final_response += response
            else:
                st.error(f"Failed to get response for chunk {i+1}")
            
            if progress_bar:
                progress_bar.progress((i + 1) / len(chunks))
            
            time.sleep(1)
    else:
        combined_prompt = text + prompt
        response = get_response(combined_prompt, provider_choice)
        if response is not None:
            final_response = response
        else:
            st.error("Failed to get response")

    if final_response:
        write_response_to_file(final_response)
        return final_response
    else:
        st.error("No valid response was generated")
        return None
    
# Streamlit app UI
st.title("Text Processor with Generative AI")

# Select provider
provider_choice = st.selectbox(
    "Choose a provider",
    ["OpenAI", "Anthropic", "Gemini"]
)

# Prefill the API key field from environment variables
api_key = st.text_input("API Key", type="password", value={
    "OpenAI": OPENAI_API_KEY,
    "Anthropic": ANTHROPIC_API_KEY,
    "Gemini": GEMINI_API_KEY
}.get(provider_choice, ""))

# Button to configure the provider
if st.button("Configure"):
    if api_key:
        if provider_choice == "OpenAI":
            configure_openai(api_key)
        elif provider_choice == "Anthropic":
            configure_anthropic(api_key)
        elif provider_choice == "Gemini":
            configure_gemini(api_key)
        st.success(f"{provider_choice} configured successfully!")
    else:
        st.error("Please enter an API key.")

# Editable prompt area
prompt_text = load_prompt()
edited_prompt = st.text_area("Edit your prompt template", value=prompt_text, height=200)

# Button to save the edited prompt
if st.button("Save Prompt"):
    save_prompt("prompt.txt", edited_prompt)
    st.success("Prompt saved successfully!")

# File uploader
uploaded_file = st.file_uploader("Upload an input text file", type="txt")

# Editable input field prefilled with the prompt from the file
if uploaded_file is not None:
    uploaded_text = uploaded_file.read().decode('utf-8')
    edited_text = st.text_area("Edit your input text", value=uploaded_text, height=300)
else:
    edited_text = ""

# Input field for chunk size (words)
chunk_size_input = st.number_input("Set chunk size (in words)", min_value=100, max_value=5000, value=500)

# Button to process the file
if st.button("Process Text"):
    if edited_text:
        # Save edited text to file
        with open("input.txt", "w") as f:
            f.write(edited_text)
        
        # Progress bar
        progress_bar = st.progress(0)
        
        # Process the text
        response_text = process_text("input.txt", provider_choice, edited_prompt, chunk_size_input, progress_bar)
        
        st.success("Processing completed successfully!")
        
        # Show the output in a large text area with buttons
        st.text_area("Output", value=response_text, height=300)

        st.download_button(
            label="Download Output",
            data=response_text,
            file_name="output.txt",
            mime="text/plain"
        )
    else:
        st.error("Please upload a file to process.")
