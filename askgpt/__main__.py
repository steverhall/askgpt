import asyncio
import json
import os
import sys
import time
from openai import AsyncOpenAI
from rich.console import Console
from rich.text import Text
import subprocess
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import var
from textual.scroll_view import ScrollView
from textual.widgets import Static, Input, Markdown, Footer
from textual.binding import Binding

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
console = Console()

DEFAULT_SYSTEM_PROMPTS=[{"name": "Linux AI", "model": "gpt-4-turbo",
                         "display_prompt": False, "content": """
           You are a helpful Linux and MacOS CLI assistant. Respond only with
           the command to execute preceded by 'CMD:'. You also know azure CLI
           and can respond to az CLI related commands. If you do not know the
           answer, or it is unrelated to Linux or MacOS, respond with 'NULL'
           """},
           {"name": "General AI", "model": "gpt-4", "content": """
           You are a helpful AI assistant. Respond as accurately as possible.
           If you are unsure of the answer, provide your best guess, but let me
           know that it is a guess. Output text as markdown.
           """}]

HOME_DIR = os.environ["HOME"]
PROMPT_FILE = os.path.join(HOME_DIR, ".ask_ai_prompts")

def load_system_prompts():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return json.load(f)
    else:
        with open(PROMPT_FILE, "w") as f:
            json.dump(DEFAULT_SYSTEM_PROMPTS, f)
            return DEFAULT_SYSTEM_PROMPTS

async def query_chatgpt(prompt, system_prompt):
    msg_history = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=msg_history)

    return response.choices[0].message.content

class OutConsole(ScrollView):
    prev = Text("")

    async def eval(self, text_input):
         with console.capture() as capture:
             try:
                 console.print(eval(text_input))
             except Exception:
                 console.print_exception(show_locals=True)
         self.prev.append(Text.from_ansi(capture.get() + "\n"))
         await self.update(self.prev)

class InConsole(Input):
    def __init__(self, out):
        super(InConsole, self).__init__()
        self.out = out

    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            await self.out.eval(self.value)
            self.value = ""


# NEED TO FIGURE OUT HOW TO PUT output and in_put
class GridTest(App):
    async def on_mount(self) -> None:
        output = OutConsole()
        in_put = InConsole(out=output)

        grid = await self.view.dock_grid(edge="left", name="left")
        grid.add_column(fraction=1, name="u")
        grid.add_row(fraction=1, name="top", min_size=3)
        grid.add_row(fraction=20, name="middle")
        grid.add_row(fraction=1, name="bottom", min_size=3)
        grid.add_areas(area1="u,top", area2="u,middle", area3="u,bottom")
        grid.place(area1=Header(), area2=output, area3=in_put,)

class AskGPT(App):

    prompt = ""
    system_prompts = []
    history = []
    BINDINGS = [Binding("ctrl+a", "change_prompt_index", "Change prompt index",
                 priority=True)]
    prompt_idx = 0
    output = OutConsole()
    in_put = InConsole(out=output)


    def compose(self) -> ComposeResult:
        yield Static("", id="prompt_name")
        yield Static("", id="prompt")
        yield Input(self.prompt, id="gpt_query",
                    placeholder="Enter request (empty to exit): ")
        yield Input("", id="command_to_exec", disabled=True)
        yield Markdown("", id="gpt_response")
        yield Footer()

    def update_display(self):
        try:
            prompt = self.system_prompts[self.prompt_idx]
            self.query_one("#prompt_name").update(prompt["name"])

            if "display_prompt" not in prompt or prompt["display_prompt"] == True:
                self.query_one("#prompt").update(prompt["content"])
                self.query_one("#prompt").display = True
            else:
                self.query_one("#prompt").display = False
        except:
            self.query_one("#prompt_name").update("Content error in .ask_ai_prompts")

    async def on_input_submitted(self, event: Input.Submitted) -> None:

        if event.input.id == "gpt_query":
            self.query_one("#gpt_query").loading = True
            result = await query_chatgpt(event.value,
                                    self.system_prompts[self.prompt_idx]["content"])

            self.query_one("#gpt_query").clear()
            self.query_one("#gpt_query").loading = False

            if (result[0:4] == "CMD:"):
                self.query_one("#command_to_exec").value = result[4:]
                self.query_one("#command_to_exec").disabled = False
                self.query_one("#gpt_query").disabled = True
                self.query_one("#command_to_exec").focus()
            else:
                self.query_one("#gpt_response").update(result)
        elif event.input.id == "command_to_exec":
            try:
                output = subprocess.check_output(event.value.split(), stderr=subprocess.STDOUT)
                self.command_output = output.decode("utf-8")
                self.query_one("#gpt_response").update("```" + self.command_output + "```")
                self.query_one("#command_to_exec").value = ""
                self.query_one("#command_to_exec").disabled = True
            except:
                self.query_one("#gpt_response").update("Error executing command")

    async def on_load(self):
        self.system_prompts = load_system_prompts()

    async def on_mount(self):
        self.update_display()

    """
    Cycle through the system prompts
    """
    def action_change_prompt_index(self):
        self.prompt_idx += 1
        if self.prompt_idx >= len(self.system_prompts):
            self.prompt_idx = 0
        self.query_one("#gpt_query").clear()
        self.update_display()

def main():
    print("running")
    app = AskGPT()
    asyncio.run(app.run(inline=True))

if __name__ == "__main__":
    main()
