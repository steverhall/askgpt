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

Some requests may require quoting the entire string, for instance, when using some punctionation (?,.$). To avoid this, just type `ask` or `ai` followed by ENTER. Then type string.

Example:

```
zsh> ai
ai: what are quotes(") for?
```

## Installation

1. Clone repository
2. `pip install openai`
3. Edit your ~.zshrc file and add the following, making sure you alter the path as appropriate for the location of your files:
4. Restart your shell, or logout and back in again


```
OPENAI_API_KEY=[YOUR OPENAI API KEY]
export OPENAI_API_KEY

ASKGPT_PATH=~/Dev/askgpt

ask() {
    if [ $# -eq 0 ]; then
        echo "ask: \c" 
        read user_input
        set -- $user_input
    fi

    local cmd=$(source $ASKGPT_PATH/venv/bin/activate && $ASKGPT_PATH/venv/bin/python $ASKGPT_PATH/ask.py "$@" && deactivate)
    # if cmd starts with ASK, then it's a question
    if [[ $cmd == ASK* ]]; then
        cmd=${cmd:5}
        print $cmd
    else
        print -z $cmd
    fi
}
ai() {
    if [ $# -eq 0 ]; then
        echo "ai: \c" 
        read user_input
        set -- $user_input
    fi

    source $ASKGPT_PATH/venv/bin/activate
    $ASKGPT_PATH/venv/bin/python $ASKGPT_PATH/ask.py -q "$@"
    deactivate
}
```

