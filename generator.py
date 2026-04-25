import os
from huggingface_hub import InferenceClient

# Your Read Token 
HF_TOKEN = os.getenv("TOKEN")

# Initialize the client with your API token
client = InferenceClient(api_key=HF_TOKEN)

async def generateAnswer(prompt: str) -> str:
    # Example for text generation
    messages = [{"role": "system", "content": prompt}]
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3", 
        messages=messages, 
        max_tokens=60,
        temperature=0.0
    )
    return completion.choices[0].message