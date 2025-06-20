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

def get_sessions_dir():
    """Get the sessions directory."""
    sessions_dir = get_config_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir

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

def create_new_session():
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    session_file = get_sessions_dir() / f"{session_id}.json"
    
    session_data = {
        "id": session_id,
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    
    return session_id

def load_session(session_id):
    """Load session data by ID."""
    session_file = get_sessions_dir() / f"{session_id}.json"
    if session_file.exists():
        try:
            with open(session_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_session(session_data):
    """Save session data."""
    session_id = session_data["id"]
    session_file = get_sessions_dir() / f"{session_id}.json"
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

def get_current_session_id():
    """Get the current session ID from config."""
    config = load_config()
    return config.get("current_session")

def set_current_session_id(session_id):
    """Set the current session ID in config."""
    config = load_config()
    config["current_session"] = session_id
    save_config(config)

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

def query_chatgpt_streaming(msg_history, model, temperature=0.7):
    api_key = validate_openai_api_key()

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
    
    return markdown_content


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
        help="Specify the OpenAI model to use (default: gpt-4o-mini or from config)."
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=None,
        help="Set the temperature for AI responses (0.0 to 2.0, default: 0.7 or from config)."
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
    parser.add_argument(
        "--new-session",
        "-n",
        action="store_true",
        help="Start a new AI conversation session (only used with --ai flag)."
    )
    return parser, parser.parse_args()

def main():
    parser, args = parse_args()
    
    # Check if no prompt was provided
    if args.prompt is None:
        parser.print_help()
        exit(1)
    
    prompt = args.prompt
    system_prompt = markdown_system_prompt if args.ai else args.system_prompt
    
    # Load config for model and temperature defaults
    config = load_config()
    
    # Use CLI values if provided, otherwise fall back to config, then hardcoded defaults
    model = args.model if args.model is not None else config.get("model", "gpt-4o-mini")
    temperature = args.temperature if args.temperature is not None else config.get("temperature", 0.7)

    # Run the async query
    if args.ai:
        # Handle session management for AI mode
        if args.new_session:
            # Create a new session
            session_id = create_new_session()
            set_current_session_id(session_id)
        else:
            # Try to load existing session
            session_id = get_current_session_id()
            if not session_id:
                # No current session, create one
                session_id = create_new_session()
                set_current_session_id(session_id)
        
        # Load session data
        session_data = load_session(session_id)
        if not session_data:
            # Session file corrupted or missing, create new one
            session_id = create_new_session()
            set_current_session_id(session_id)
            session_data = load_session(session_id)
        
        # Build message history
        msg_history = []
        
        # Add system prompt if this is the first message in session
        if not session_data["messages"]:
            msg_history.append({"role": "system", "content": system_prompt})
        else:
            # Load existing messages from session
            msg_history.extend(session_data["messages"])
        
        # Add new user message
        msg_history.append({"role": "user", "content": prompt})
        
        # Get AI response
        ai_response = query_chatgpt_streaming(msg_history, model, temperature)
        
        # Add AI response to message history
        msg_history.append({"role": "assistant", "content": ai_response})
        
        # Update session data (only store messages, not system prompt for ongoing conversations)
        if not session_data["messages"]:
            # First interaction - store system prompt
            session_data["messages"] = msg_history
        else:
            # Add new user and assistant messages
            session_data["messages"].extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ai_response}
            ])
        
        # Save updated session
        save_session(session_data)
    else:
        response = asyncio.run(query_chatgpt(prompt, system_prompt, model, temperature))
        console.print(response)

if __name__ == "__main__":
    main()
