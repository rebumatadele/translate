import streamlit as st
import openai
from anthropic import Anthropic
import google.generativeai as genai
import time
from dotenv import load_dotenv
import os
from curl_cffi import requests

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
def generate_with_openai(prompt, model="gpt-4"):
    response = openai.chat.completions.create(
        model= model,  # Use GPT-4 model
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Function to configure Anthropic (Claude)
def configure_anthropic(api_key):
    global ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = api_key

def generate_with_anthropic(prompt):
    headers = {
        'x-api-key': api_key,
        'content-type': 'application/json',
        'anthropic-version': '2023-06-01',
        "anthropic-dangerous-direct-browser-access": "true",
    }

    data = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,

    }
    try:
        response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data, timeout=30)
        
        # Handle the response
        if response.status_code == 200:
            response_json = response.json()

            # Extract text from the content field inside the response
            if "content" in response_json and isinstance(response_json["content"], list):
                return "".join([item.get("text", "") for item in response_json["content"]])
            else:
                return "No content field in response"
        else:
            error_message = f"Anthropic Error: {response.status_code} - {response.json().get('error', {}).get('message', 'Unknown error')}"
            st.error(error_message)
            return None
    except Exception as e:
        error_message = f"An error occurred: {e}"
        st.error(error_message)
        print(error_message)
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
def process_text(text_or_file, provider_choice, prompt, chunk_size_in_words, model_choice=None, progress_bar=None):
    text = load_text(text_or_file)
    if not text:
        return
    
    final_response = ""
    
    if provider_choice in ["OpenAI", "Anthropic"]:
        chunks = split_text_into_chunks(text, chunk_size_in_words)

        for i, chunk in enumerate(chunks):
            combined_prompt = prompt + chunk 
            response = get_response(combined_prompt, provider_choice, model_choice=model_choice)
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

# Add model selection for OpenAI
model_choice = None
if provider_choice == "OpenAI":
    model_choice = st.selectbox(
        "Choose a model",
        ["gpt-3.5-turbo", "gpt-4"]
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
        response_text = process_text("input.txt", provider_choice, edited_prompt, chunk_size_input, model_choice=model_choice, progress_bar=progress_bar)
        
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
