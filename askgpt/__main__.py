import argparse
import asyncio
import os
from openai import AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown

console = Console()
default_system_prompt = "You are an assistant that answers with a single MacOS CLI command based on the request. No other output, it must be a valid MacOS CLI command and if none can be given, respond with NULL. Again, only output the command, no markdown, no additional words or help."
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
    response = asyncio.run(query_chatgpt(prompt, system_prompt, model))
    if args.ai:
        console.print(Markdown(response))
    else:
        console.print(response)

if __name__ == "__main__":
    main()