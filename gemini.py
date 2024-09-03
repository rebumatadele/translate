import streamlit as st
import google.generativeai as genai
import os

# Configure the Generative AI API
genai.configure(api_key="AIzaSyARFySyhjCOD4VLh0r6TB_EOy1CTTk7TaA")

# Dropdown to select the model
model_choice = st.selectbox(
    "Choose a model",
    ["Gemini", "Claude", "OpenAI"]
)

# Configure the selected model
def get_model(model_choice):
    if model_choice == "Gemini":
        return genai.GenerativeModel("gemini-1.5-flash")
    elif model_choice == "Claude":
        return genai.GenerativeModel("claude-v1")  # Example, replace with correct identifier
    elif model_choice == "OpenAI":
        return genai.GenerativeModel("openai-gpt-3.5")  # Example, replace with correct identifier

model = get_model(model_choice)

# Load text from a file
def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Write response to a file
def write_response_to_file(response_text, file_name="output.txt"):
    with open(file_name, 'w') as file:
        file.write(response_text)

# Process the text and save to output file
def process_text(text_or_file, model):
    text = load_text(text_or_file)
    prompt = text + load_text("prompt.txt")
    response = model.generate_content(prompt)
    write_response_to_file(response.text)

# Streamlit app UI
st.title("Text Processor with Generative AI")

# File uploader
uploaded_file = st.file_uploader("Upload an input text file", type="txt")

# Button to process the file
if st.button("Process File"):
    if uploaded_file is not None:
        # Save uploaded file
        with open("input.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the file
        process_text("input.txt", model)
        
        st.success("File processed successfully!")
        
        # Provide download link for the output file
        with open("output.txt", "rb") as file:
            btn = st.download_button(
                label="Download Output",
                data=file,
                file_name="output.txt",
                mime="text/plain"
            )
    else:
        st.error("Please upload a file to process.")
