import json
import os
import sys
import time
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
system_prompts=[{"role": "system", "content": """
           You are a helpful Linux CLI assistant. Respond only with
           the command to execute and nothing else. You also know azure cli
           and can respond to az CLI related commands.
           To non-linux related questions, respond as normal, but prefix
           response with 'ASK:', this includes code generation responses.
           """},
          {"role": "system", "content": """
           You are a helpful AI assistant. Respond as accurately as possible.
           If you are unsure of the answer, provide your best guess, but let me
           know that it is a guess. Output text as markdown.
           """}]

MAX_HISTORY_AGE = 120

# Ensure there is an argument provided
if len(sys.argv) < 2:
    print("Usage: python ask.py 'your_prompt_here'")
    sys.exit(1)

def save_history(messages, ai_reponse):
    messages.append({"role": "assistant", "content": ai_reponse})
    filename = os.path.join(os.environ["HOME"], ".ask_ai")
    with open(filename, "w") as f:
        json.dump(messages, f)

def read_history(newrequest, systemprompt):
    filename = os.path.join(os.environ["HOME"], ".ask_ai")
    history = []
    if os.path.exists(filename):
        if time.time() - os.path.getmtime(filename) < MAX_HISTORY_AGE:
            with open(filename, "r") as f:
                history = json.load(f)
        else:
            os.remove(filename)

    if len(history) == 0:
        history=[systemprompt, {"role": "user", "content": newrequest}]
    else:
        history.append({"role": "user", "content": newrequest})

    return history

def query_chatgpt(prompt, system_prompt):
    response = client.chat.completions.create(model="gpt-4",
    messages=[system_prompt,
              {"role": "user", "content": prompt}])
    return response.choices[0].message.content


def markdown_chatgpt(prompt, system_prompt):
    console = Console()
    msg_history = read_history(prompt, system_prompt)
    with Progress() as progress:
        task = progress.add_task("[green]Processing...", total=None)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=msg_history)
    save_history(msg_history, response.choices[0].message.content)


    console.print(Markdown(response.choices[0].message.content))


if __name__ == "__main__":
    # check if -q argument is provided
    if sys.argv[1] == "-q":
        # remove -q argument
        sys.argv.pop(0)
        prompt = ""
        for arg in sys.argv[1:]:
            prompt += " " + arg
        markdown_chatgpt(prompt, system_prompts[1])
    else:
        prompt = ""
        for arg in sys.argv[1:]:
            prompt += " " + arg

        response = query_chatgpt(prompt, system_prompts[0])
        print(response)

