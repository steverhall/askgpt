import argparse
import asyncio
import os
from openai import AsyncOpenAI, OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()
default_system_prompt = "You are an assistant that answers with a single Ubuntu Linux CLI command based on the request. No other output, it must be a valid Ubuntu Linux CLI command and if none can be given, respond with NULL. Again, only output the command, no markdown, no additional words or help."
markdown_system_prompt = "You are an assistant that responds with output formatted in Markdown."

async def query_chatgpt(prompt, system_prompt, model):
    if system_prompt == "":
        system_prompt = default_system_prompt
    msg_history = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}]

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")    )

    response = await client.chat.completions.create(
        messages=msg_history,
        model=model
    )

    return response.choices[0].message.content

def query_chatgpt_streaming(prompt, system_prompt, model):
    if system_prompt == "":
        system_prompt = default_system_prompt
    msg_history = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}]

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # print messages as it arrives
    response = client.chat.completions.create(
        messages=msg_history,
        model=model,
        stream=True
    )

    markdown_content = ""

    # Process each chunk as it arrives
    with Live(refresh_per_second=8) as live:
        for chunk in response:
            new_content = chunk.choices[0].delta.content
            # check if new_content is a string
            if isinstance(new_content, str):
                markdown_content += new_content
                live.update(Markdown(markdown_content))


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for AI Assistant")

    parser.add_argument(
        "--system-prompt",
        "-s",
        type=str,
        default="",
        help="Custom system prompt for the AI. If omitted, a default will be used."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        help="Specify the OpenAI model to use (default: gpt-4)."
    )
    parser.add_argument(
        "--ai",
        "-a",
        action="store_true",
        help="Use a specialized system prompt to format output as Markdown."
    )
    # Add prompt arg, allowing for a custom unquoted string

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt to send to the AI."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    prompt = args.prompt
    system_prompt = markdown_system_prompt if args.ai else args.system_prompt
    model = args.model

    # Run the async query
    if args.ai:
        query_chatgpt_streaming(prompt, system_prompt, model)
    else:
        response = asyncio.run(query_chatgpt(prompt, system_prompt, model))
        console.print(response)

if __name__ == "__main__":
    main()
