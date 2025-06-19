import argparse
import asyncio
import json
import os
import toml
import uuid
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI, OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()
default_system_prompt = "You are an assistant that answers with a single Ubuntu Linux CLI command based on the request. No other output, it must be a valid Ubuntu Linux CLI command and if none can be given, respond with NULL. Again, only output the command, no markdown, no additional words or help."
markdown_system_prompt = "You are an assistant that responds with output formatted in Markdown."

def validate_openai_api_key():
    """Validate that OPENAI_API_KEY environment variable is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable is not set.[/red]")
        console.print()
        console.print("Please set your OpenAI API key as an environment variable:")
        console.print("  export OPENAI_API_KEY='your-api-key-here'")
        console.print()
        console.print("You can also add it to your shell configuration file (e.g., ~/.zshrc or ~/.bashrc):")
        console.print("  OPENAI_API_KEY=your-api-key-here")
        console.print("  export OPENAI_API_KEY")
        console.print()
        console.print("Get your API key from: https://platform.openai.com/api-keys")
        exit(1)
    return api_key

def get_config_dir():
    """Get the askgpt configuration directory."""
    config_dir = Path.home() / ".config" / "askgpt"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def get_config_file():
    """Get the configuration file path."""
    return get_config_dir() / "askgpt.toml"

def load_config():
    """Load configuration from askgpt.toml."""
    config_file = get_config_file()
    if config_file.exists():
        try:
            return toml.load(config_file)
        except Exception:
            return {}
    return {}

def save_config(config):
    """Save configuration to askgpt.toml."""
    config_file = get_config_file()
    with open(config_file, "w") as f:
        toml.dump(config, f)

def get_default_config():
    """Get default configuration values."""
    return {
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }

async def query_chatgpt(prompt, system_prompt, model, temperature=0.7):
    api_key = validate_openai_api_key()
    
    if system_prompt == "":
        system_prompt = default_system_prompt
    msg_history = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}]

    client = AsyncOpenAI(
        api_key=api_key    )

    response = await client.chat.completions.create(
        messages=msg_history,
        model=model,
        temperature=temperature
    )

    return response.choices[0].message.content

def query_chatgpt_streaming(prompt, system_prompt, model, temperature=0.7):
    api_key = validate_openai_api_key()
    
    if system_prompt == "":
        system_prompt = default_system_prompt
    msg_history = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}]

    client = OpenAI(
        api_key=api_key
    )

    # print messages as it arrives
    response = client.chat.completions.create(
        messages=msg_history,
        model=model,
        temperature=temperature,
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
        default=None,
        help="Specify the OpenAI model to use (default from config: gpt-4o-mini)."
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=None,
        help="Temperature for the AI response (0.0-2.0, default from config: 0.7)."
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
    
    # Load configuration and apply defaults
    config = load_config()
    defaults = get_default_config()
    
    # Create config file with defaults if it doesn't exist
    if not get_config_file().exists():
        save_config(defaults)
        config = defaults
    else:
        # Ensure config has all required keys with defaults
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        save_config(config)
    
    prompt = args.prompt
    system_prompt = markdown_system_prompt if args.ai else args.system_prompt
    model = args.model if args.model is not None else config.get("model", defaults["model"])
    temperature = args.temperature if args.temperature is not None else config.get("temperature", defaults["temperature"])

    # Run the async query
    if args.ai:
        query_chatgpt_streaming(prompt, system_prompt, model, temperature)
    else:
        response = asyncio.run(query_chatgpt(prompt, system_prompt, model, temperature))
        console.print(response)

if __name__ == "__main__":
    main()
