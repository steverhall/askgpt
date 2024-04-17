import asyncio
import json
import os
import sys
import time
from openai import AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import var
from textual.widgets import Static, Input, Markdown, Footer
from textual.binding import Binding

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_SYSTEM_PROMPTS=[{"name": "Linux AI", "content": """
           You are a helpful Linux and MacOS CLI assistant. Respond only with
           the command to execute and nothing else. You also know azure CLI
           and can respond to az CLI related commands. If you do not know the
           answer, or it is unrelated to Linux or MacOS, respond with 'NULL'
           """},
          {"name": "General AI", "content": """
           You are a helpful AI assistant. Respond as accurately as possible.
           If you are unsure of the answer, provide your best guess, but let me
           know that it is a guess. Output text as markdown.
           """}]

MAX_HISTORY_AGE = 120
HOME_DIR = os.environ["HOME"]
PROMPT_FILE = os.path.join(HOME_DIR, ".ask_ai_prompts")
HISTORY_FILE = os.path.join(HOME_DIR, ".ask_ai_history")

def load_system_prompts():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return json.load(f)
    else:
        with open(PROMPT_FILE, "w") as f:
            json.dump(DEFAULT_SYSTEM_PROMPTS, f)
            return DEFAULT_SYSTEM_PROMPTS

def save_history(messages, ai_reponse):
    messages.append({"role": "assistant", "content": ai_reponse})
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f)

def read_history(newrequest, systemprompt):
    history = []
    if os.path.exists(HISTORY_FILE):
        if time.time() - os.path.getmtime(HISTORY_FILE) < MAX_HISTORY_AGE:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            os.remove(HISTORY_FILE)

    if len(history) == 0:
        history=[{"role": "system", "content": systemprompt}, {"role": "user", "content": newrequest}]
    else:
        history.append({"role": "user", "content": newrequest})

    return history

def query_chatgpt(prompt, system_prompt):
    response = client.chat.completions.create(model="gpt-4",
    messages=[system_prompt,
              {"role": "user", "content": prompt}])
    return response.choices[0].message.content


async def markdown_chatgpt(prompt, system_prompt):
    console = Console()
    msg_history = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=msg_history)

    return response.choices[0].message.content


class AskGPT(App):

    prompt = ""
    system_prompts = []
    history = []
    BINDINGS = [Binding("ctrl+a", "toggle_query_type", "Toggle query type",
                 priority=True)]
    is_general_ai_query = False
    prompt_idx = 0


    def compose(self) -> ComposeResult:
        yield Input(self.prompt, id="gpt_query",
                    placeholder="Enter request (empty to exit): ")
        yield Static("", id="prompt")
        yield Markdown("", id="gpt_response")
        yield Footer()

    def header_text(self):
        CTRL_A = " (Ctrl+A to toggle)"
        return self.system_prompts[self.prompt_idx]["name"] + CTRL_A

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value == "":
            sys.exit()

        self.query_one("#gpt_query").loading = True
        self.query_one("#gpt_query").clear()
        result = await markdown_chatgpt(event.value,
                                  self.system_prompts[self.prompt_idx]["content"])
        self.query_one("#gpt_query").loading = False
        self.query_one("#gpt_response").update(result)

    async def on_load(self):
        self.system_prompts = load_system_prompts()

    async def on_mount(self):
        self.query_one("#prompt").update(self.system_prompts[self.prompt_idx]["content"])

    def action_toggle_query_type(self):
        self.prompt_idx += 1
        if self.prompt_idx >= len(self.system_prompts):
            self.prompt_idx = 0
        self.is_general_ai_query = not self.is_general_ai_query
        self.query_one("#gpt_query").clear()
        self.query_one("#prompt").update(self.system_prompts[self.prompt_idx]["content"])

def main():
    app = AskGPT()
    asyncio.run(app.run(inline=True))

