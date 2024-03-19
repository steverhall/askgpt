import os
import sys
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure there is an argument provided
if len(sys.argv) < 2:
    print("Usage: python ask.py 'your_prompt_here'")
    sys.exit(1)

# Your OpenAI API key

def query_chatgpt(prompt):
    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": """
               You are a helpful Linux CLI assistant Linux. Respond only with
               the command to execute and nothing else. You also know azure cli
               and can respond to az CLI related commands.
               To non-linux related questions, respond as normal, but prefix
               response with 'ASK:', this includes code generation responses.
               """},
              {"role": "user", "content": prompt}])
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = ""
    for arg in sys.argv[1:]:
        prompt += " " + arg

    response = query_chatgpt(prompt)
    print(response)

