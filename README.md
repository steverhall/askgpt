# askgpt
CLI ChatGPT integration for discovering Linux commands or answering random ChatGPT queries.


Askgpt can be used as a stand-alone GPT CLI or plugged-in to zsh (or other shells) in order to act as your CLI assistant.

When configured as below, you can interact with ChatGPT from the command line, and either:

1. Have it respond to a CLI-related question, putting the linux command in the queue for you to execute, or
2. Respond to a general question (non-linux related)


## Linux CLI commands

Askgpt will attempt to recognize when your query is about a linux command and, when configured properly in zsh, will allow you to execute that command by just pressing enter.

Example:

```
zsh> ask how do I list all files including hidden
zsh> ls -a  
```

To accept askgpt's response, press enter. To cancel, backspace over text or press CTRL-C.


## Standard ChatGPT requests

If the query appears to be unrelated to a linux command, askgpt will just print the results.
You can force the general response by using `ai` instead of `ask`

Example:

```
zsh> ai what is the highest mountain
The highest mountain in the world is Mount Everest.
zsh>
```

## Quoting

Some requests may require quoting the entire string, for instance, when using some punctuation (?,.$). To avoid this, just type `ask` or `ai` followed by ENTER. Then type string.

Example:

```
zsh> ai
ai: what are quotes(") for?
```

## Installation

### 1. Install the CLI

Clone the repository and install with pipx (recommended):

```sh
git clone https://github.com/steverhall/askgpt.git
cd askgpt
pipx install .
```

### 2. Enable the zsh plugin (optional, for best integration)

Copy or symlink `askgpt.plugin.zsh` to a directory in your fpath, or source it directly in your `.zshrc`:

```sh
# Add this to your ~/.zshrc
source /path/to/askgpt/askgpt.plugin.zsh
```

#### Oh-My-Zsh

Clone or symlink the plugin into your custom plugins directory, then add `askgpt` to your plugins list in `.zshrc`:

```sh
git clone https://github.com/steverhall/askgpt.git $ZSH_CUSTOM/plugins/askgpt
# In your .zshrc:
plugins=(... askgpt)
```

#### zinit

Add this to your `.zshrc`:

```sh
zinit light steverhall/askgpt
```

#### Windows

```powershell
if (!(Test-Path -Path $PROFILE)) {
    New-Item -Type File -Path $PROFILE -Force
}
notepad $PROFILE
```

Add the following contents to your PowerShell profile:

```powershell
function ask {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Prompt
    )
    if (-not $Prompt) {
        $Prompt = Read-Host "ask"
    } else {
        $Prompt = $Prompt -join " "
    }
    $cmd = askgpt --prompt "$Prompt"
    if ($cmd -eq "NULL") {
        Write-Host "No results found"
    } else {
        # Place command in command line for review (simulate with clipboard)
        Set-Clipboard $cmd
        Write-Host "Command copied to clipboard. Paste and press Enter to run."
    }
}

function ai {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Prompt
    )
    if (-not $Prompt) {
        $Prompt = Read-Host "ai"
    } else {
        $Prompt = $Prompt -join " "
    }
    askgpt --ai --prompt "$Prompt"
}
```

Then, as admin:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. Set your OpenAI API key

Edit `askgpt.plugin.zsh` or set in your `.zshrc`:

```sh
export OPENAI_API_KEY=[YOUR OPENAI API KEY]
```

## Configuration

Askgpt supports configuration through a TOML file located at `$HOME/.config/askgpt/askgpt.toml`.

You can set default values for model and temperature:

```toml
model = "gpt-4"
temperature = 0.8
```

Command line arguments will override config file values:

```bash
# Uses model and temperature from config file
askgpt --prompt "Hello"

# Overrides config file model
askgpt --model "gpt-3.5-turbo" --prompt "Hello"

# Overrides config file temperature  
askgpt --temperature 0.3 --prompt "Hello"
```


## Uninstall

```sh
pipx uninstall askgpt
```

## Shell Functions (for reference)

The plugin provides these functions:

```zsh
ask  # Query for a CLI command and queue it for execution
ai   # Query for a general answer
```
