import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Function to configure OpenAI
def configure_openai(api_key):
    openai.api_key = api_key

# Function to generate response with OpenAI
def generate_with_openai(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content

# Function to configure Anthropic (Claude)
def configure_anthropic(api_key):
    anthropic.api_key = api_key

# Function to generate response with Anthropic (Claude)
def generate_with_anthropic(prompt):
    client = anthropic.Client(api_key=anthropic.api_key)
    response = client.completions.create(
        model="claude-1",  # Specify the model, adjust if needed
        prompt=f"\n\nHuman: {prompt}\n\nAssistant:",  # Anthropic uses specific prompt formatting
        max_tokens_to_sample=150,
        stop_sequences=["\n\nHuman:"],  # This is optional but useful to prevent long responses
        temperature=0.7  # Optional, adjust as needed
    )
    return response['completion']


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
def process_text(text_or_file, provider_choice, progress_bar=None):
    text = load_text(text_or_file)
    if not text:
        return
    
    final_response = ""
    
    if provider_choice in ["OpenAI", "Anthropic"]:
        # Define the chunk size (adjust based on needs)
        chunk_size = 3000  # Adjust the chunk size to fit within the token limit

        # Split text into chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Iterate over each chunk, send request, and append the response
        for i, chunk in enumerate(chunks):
            prompt = chunk + load_text("prompt.txt")  # Add any additional prompt text if needed
            response = get_response(prompt, provider_choice)
            final_response += response
            
            # Update progress bar
            if progress_bar:
                progress_bar.progress((i + 1) / len(chunks))
            
            # Delay between requests to avoid hitting rate limits
            time.sleep(1)  # 1-second delay between requests
    else:
        # For Gemini, process the entire text at once
        prompt = text + load_text("prompt.txt")  # Add any additional prompt text if needed
        final_response = get_response(prompt, provider_choice)

    # Write the final accumulated response to the output file
    write_response_to_file(final_response)
    return final_response

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

# File uploader
uploaded_file = st.file_uploader("Upload an input text file", type="txt")

# Editable input field prefilled with the prompt from the file
if uploaded_file is not None:
    uploaded_text = uploaded_file.read().decode('utf-8')
    edited_text = st.text_area("Edit your prompt", value=uploaded_text, height=300)
else:
    edited_text = ""

# Button to process the file
if st.button("Process Text"):
    if edited_text:
        # Save edited text to file
        with open("input.txt", "w") as f:
            f.write(edited_text)
        
        # Progress bar
        progress_bar = st.progress(0)
        
        # Process the text
        response_text = process_text("input.txt", provider_choice, progress_bar)
        
        st.success("Processing completed successfully!")
        
        # Show the output in a large text area with buttons
        st.text_area("Output", value=response_text, height=300)
        

        st.download_button(
            label="Download Output",
            data=response_text,
            file_name="output.txt",
            mime="text/plain"
            )

        # Display the copied text in the session state
        if "copied_text" in st.session_state:
            st.code(st.session_state.copied_text)
    else:
        st.error("Please upload a file to process.")
