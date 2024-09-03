import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import backoff
import time
# Store API key securely
openai.api_key = "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"

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_to_file(responses, output_file):
    with open(output_file, 'w') as file:
        for response in responses:
            file.write(response + '\n')

# Backoff retry strategy for handling rate limit errors
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def call_openai_api(chunk):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Say Hello: {chunk}."},
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.RateLimitError as e:
        print(f"Rate limit error: {e}. Retrying...")
        raise  # This will trigger the backoff retry
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(encoding.decode(words[i:i + tokens]))
        time.sleep(10)
    return chunks   

def process_chunks(input_file, output_file):
    text = load_text(input_file)
    chunks = split_into_chunks(text)
    
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))

    # Filter out None responses if any API calls failed
    responses = [r for r in responses if r is not None]

    save_to_file(responses, output_file)

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    process_chunks(input_file, output_file)
