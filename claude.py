import anthropic
from concurrent.futures import ThreadPoolExecutor
import backoff

# Store API key securely
client = anthropic.Client(api_key="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")

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_to_file(responses, output_file):
    with open(output_file, 'w') as file:
        for response in responses:
            file.write(response + '\n')

# Backoff retry strategy for handling rate limit errors
@backoff.on_exception(backoff.expo, anthropic.AnthropicException, max_tries=5)
def call_claude_api(chunk):
    try:
        response = client.completions.create(
            model="claude-2",
            prompt=f"{anthropic.HUMAN_PROMPT} Say Hello: {chunk}. {anthropic.AI_PROMPT}",
            max_tokens_to_sample=500,
            temperature=0.5,
        )
        return response['completion'].strip()
    except anthropic.AnthropicException as e:
        print(f"Rate limit error or other API error: {e}. Retrying...")
        raise  # This will trigger the backoff retry
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def split_into_chunks(text, tokens=500):
    # Claude does not have a specific tokenizer like OpenAI's tiktoken, so you might want to split based on characters
    chunks = [text[i:i + tokens] for i in range(0, len(text), tokens)]
    return chunks   

def process_chunks(input_file, output_file):
    text = load_text(input_file)
    chunks = split_into_chunks(text)
    
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_claude_api, chunks))

    # Filter out None responses if any API calls failed
    responses = [r for r in responses if r is not None]

    save_to_file(responses, output_file)

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    process_chunks(input_file, output_file)
