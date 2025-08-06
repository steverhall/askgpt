# askgpt.plugin.zsh
# Zsh plugin for askgpt CLI integration

# Ensure OPENAI_API_KEY is set (user should edit this line)
if [[ -z "$OPENAI_API_KEY" ]]; then
  export OPENAI_API_KEY="[YOUR OPENAI API KEY]"
fi

# ask function: get a CLI command from askgpt and queue it for execution
ask() {
    if [ $# -eq 0 ]; then
        echo -n "ask: "
        read user_input
        set -- $user_input
    fi
    local prompt=$(printf "%s " "$@")
    local cmd=$(askgpt --prompt "$prompt")
    if [[ $cmd == NULL ]]; then
        print "No results found"
    else
        print -z $cmd
    fi
}

# ai function: get a general answer from askgpt
ai() {
    if [ $# -eq 0 ]; then
        echo -n "ai: "
        read user_input
        set -- $user_input
    fi
    local prompt=$(printf "%s " "$@")
    askgpt --ai --prompt "$prompt"
}

# Optional: autoload functions
autoload -Uz ask ai
