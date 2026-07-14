from dotenv import load_dotenv
import os
from openai import OpenAI

# Load variables from .env
load_dotenv()

# Read API key from .env
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

prompt = "Explain Artificial Intelligence in simple words."

temperatures = [0.0, 0.3, 0.7, 1.0]

for temp in temperatures:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
        max_tokens=300
    )

    print("=" * 50)
    print(f"Temperature: {temp}")
    print("=" * 50)
    print(response.choices[0].message.content)
    print()