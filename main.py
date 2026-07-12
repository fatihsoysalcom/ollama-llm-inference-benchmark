import urllib.request
import json
import time
import sys

# Configuration for Ollama server and model
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama2" # Or any other model you have pulled, e.g., "mistral", "gemma:2b"
PROMPT = "Why is edge AI important?"

def run_ollama_inference_benchmark():
    """
    Connects to a local Ollama server, sends a prompt, and measures the response time.
    """
    print(f"Attempting to connect to Ollama at {OLLAMA_API_URL}...")
    print(f"Using model: {MODEL_NAME}")
    print(f"Prompt: '{PROMPT}'\n")

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "stream": False # We want the full response at once for benchmarking
    }

    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}

    req = urllib.request.Request(OLLAMA_API_URL, data=data, headers=headers, method='POST')

    try:
        start_time = time.perf_counter() # Start timing for benchmark
        with urllib.request.urlopen(req, timeout=300) as response: # 5-minute timeout
            end_time = time.perf_counter() # End timing
            response_body = response.read().decode('utf-8')
            
            # Ollama's /api/generate endpoint returns a single JSON object if stream=False
            result = json.loads(response_body)
            
            generated_text = result.get("response", "No response field found.")
            total_duration = end_time - start_time

            print("--- Ollama Inference Result ---")
            print(f"Generated Text:\n{generated_text}\n")
            print(f"Time taken for inference: {total_duration:.2f} seconds")
            print("-------------------------------\n")

            # Basic check for a valid response from Ollama
            if "error" in result:
                print(f"Error from Ollama: {result['error']}")
                sys.exit(1)
            elif not generated_text or generated_text == "No response field found.":
                print("Warning: Ollama returned an empty or invalid response.")
                sys.exit(1)

    except urllib.error.URLError as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running and the model is pulled.")
        print("You can start Ollama with `ollama serve` and pull a model with `ollama pull llama2`.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Ollama: {e}")
        print(f"Raw response body: {response_body[:500]}...")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ollama_inference_benchmark()
