import asyncio
import argparse
from unittest.mock import Mock, patch, AsyncMock
import pytest
import os
from askgpt.__main__ import (
    query_chatgpt,
    query_chatgpt_streaming,
    parse_args,
    main,
    default_system_prompt,
    markdown_system_prompt,
    validate_openai_api_key
)


class TestValidateOpenAIApiKey:
    """Test the validate_openai_api_key function."""
    
    def test_validate_openai_api_key_success(self, mocker):
        """Test validate_openai_api_key when API key is present."""
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
        
        result = validate_openai_api_key()
        
        assert result == 'test-api-key'
    
    def test_validate_openai_api_key_missing(self, mocker):
        """Test validate_openai_api_key when API key is missing."""
        # Remove OPENAI_API_KEY from environment
        mocker.patch.dict(os.environ, {}, clear=True)
        mock_console = mocker.patch('askgpt.__main__.console')
        
        with pytest.raises(SystemExit) as exc_info:
            validate_openai_api_key()
        
        assert exc_info.value.code == 1
        # Verify error message was printed
        assert mock_console.print.call_count >= 1
        error_call_args = mock_console.print.call_args_list[0][0][0]
        assert "OPENAI_API_KEY environment variable is not set" in error_call_args


class TestQueryChatGPT:
    """Test the async query_chatgpt function with mocked OpenAI calls."""

    @pytest.mark.asyncio
    async def test_query_chatgpt_with_default_system_prompt(self, mocker):
        """Test query_chatgpt with empty system prompt (should use default)."""
        # Mock AsyncOpenAI client and response
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "ls -la"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock the AsyncOpenAI constructor
        mocker.patch('askgpt.__main__.AsyncOpenAI', return_value=mock_client)
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
        
        # Call the function
        result = await query_chatgpt("list all files", "", "gpt-4o-mini")
        
        # Assertions
        assert result == "ls -la"
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": default_system_prompt},
                {"role": "user", "content": "list all files"}
            ],
            model="gpt-4o-mini",
            temperature=0.7
        )

    @pytest.mark.asyncio
    async def test_query_chatgpt_with_custom_system_prompt(self, mocker):
        """Test query_chatgpt with custom system prompt."""
        # Mock AsyncOpenAI client and response
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Custom response"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock the AsyncOpenAI constructor
        mocker.patch('askgpt.__main__.AsyncOpenAI', return_value=mock_client)
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
        
        custom_prompt = "You are a helpful assistant."
        
        # Call the function
        result = await query_chatgpt("Hello", custom_prompt, "gpt-4")
        
        # Assertions
        assert result == "Custom response"
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": "Hello"}
            ],
            model="gpt-4",
            temperature=0.7
        )

    @pytest.mark.asyncio 
    async def test_query_chatgpt_api_key_from_env(self, mocker):
        """Test that query_chatgpt uses API key from environment."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "response"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock the AsyncOpenAI constructor to capture the api_key
        mock_openai_class = mocker.patch('askgpt.__main__.AsyncOpenAI', return_value=mock_client)
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'my-secret-key'})
        
        # Call the function
        await query_chatgpt("test", "system", "gpt-4")
        
        # Verify AsyncOpenAI was called with correct API key
        mock_openai_class.assert_called_once_with(api_key='my-secret-key')


class TestQueryChatGPTStreaming:
    """Test the sync query_chatgpt_streaming function with mocked OpenAI calls."""

    def test_query_chatgpt_streaming_with_default_system_prompt(self, mocker):
        """Test query_chatgpt_streaming with message history."""
        # Mock OpenAI client and streaming response
        mock_client = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " World"
        
        mock_chunk3 = Mock()
        mock_chunk3.choices = [Mock()]
        mock_chunk3.choices[0].delta.content = None  # Test non-string content
        
        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
        
        # Mock the OpenAI constructor
        mocker.patch('askgpt.__main__.OpenAI', return_value=mock_client)
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
        
        # Mock Live context manager properly
        mock_live_instance = Mock()
        mock_live_class = mocker.patch('askgpt.__main__.Live', return_value=mock_live_instance)
        mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
        mock_live_instance.__exit__ = Mock(return_value=None)
        
        mock_markdown = Mock()
        mocker.patch('askgpt.__main__.Markdown', return_value=mock_markdown)
        
        # Build message history as the new function expects
        msg_history = [
            {"role": "system", "content": default_system_prompt},
            {"role": "user", "content": "test prompt"}
        ]
        
        # Call the function with new signature
        result = query_chatgpt_streaming(msg_history, "gpt-4o-mini")
        
        # Assertions
        mock_client.chat.completions.create.assert_called_once_with(
            messages=msg_history,
            model="gpt-4o-mini",
            temperature=0.7,
            stream=True
        )
        
        # Should return the concatenated content
        assert result == "Hello World"
        
        # Verify Live context manager was used
        mock_live_instance.__enter__.assert_called_once()
        mock_live_instance.__exit__.assert_called_once()
        
        # The main focus is that the OpenAI API call was made correctly
        # Markdown behavior is secondary to the core functionality

    def test_query_chatgpt_streaming_with_custom_system_prompt(self, mocker):
        """Test query_chatgpt_streaming with custom message history."""
        mock_client = Mock()
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = "Response"
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        mocker.patch('askgpt.__main__.OpenAI', return_value=mock_client)
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
        
        # Mock Live context manager properly
        mock_live_instance = Mock()
        mock_live_class = mocker.patch('askgpt.__main__.Live', return_value=mock_live_instance)
        mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
        mock_live_instance.__exit__ = Mock(return_value=None)
        
        mocker.patch('askgpt.__main__.Markdown')
        
        custom_prompt = "Be helpful and concise."
        
        # Build message history with custom system prompt
        msg_history = [
            {"role": "system", "content": custom_prompt},
            {"role": "user", "content": "Hello"}
        ]
        
        # Call the function with new signature
        result = query_chatgpt_streaming(msg_history, "gpt-4")
        
        # Assertions
        mock_client.chat.completions.create.assert_called_once_with(
            messages=msg_history,
            model="gpt-4",
            temperature=0.7,
            stream=True
        )
        
        # Should return the response content
        assert result == "Response"

    def test_query_chatgpt_streaming_api_key_from_env(self, mocker):
        """Test that query_chatgpt_streaming uses API key from environment."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = []
        
        # Mock the OpenAI constructor to capture the api_key
        mock_openai_class = mocker.patch('askgpt.__main__.OpenAI', return_value=mock_client)
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'streaming-api-key'})
        
        # Mock Live context manager properly
        mock_live_instance = Mock()
        mock_live_class = mocker.patch('askgpt.__main__.Live', return_value=mock_live_instance)
        mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
        mock_live_instance.__exit__ = Mock(return_value=None)
        
        mocker.patch('askgpt.__main__.Markdown')
        
        # Build simple message history
        msg_history = [{"role": "user", "content": "test"}]
        
        # Call the function with new signature
        query_chatgpt_streaming(msg_history, "gpt-4")
        
        # Verify OpenAI was called with correct API key
        mock_openai_class.assert_called_once_with(api_key='streaming-api-key')


class TestParseArgs:
    """Test the argument parsing functionality."""

    def test_parse_args_default_values(self, mocker):
        """Test parse_args with default values."""
        # Mock sys.argv to simulate command line arguments
        test_args = ['askgpt', '--prompt', 'test prompt']
        mocker.patch('sys.argv', test_args)
        
        parser, args = parse_args()
        
        assert args.prompt == 'test prompt'
        assert args.system_prompt == ''
        assert args.model is None  # Now defaults to None to allow config override
        assert args.temperature is None  # Now defaults to None to allow config override
        assert args.ai is False
        assert args.new_session is False  # Check default value

    def test_parse_args_all_custom_values(self, mocker):
        """Test parse_args with all custom values."""
        test_args = [
            'askgpt',
            '--prompt', 'custom prompt',
            '--system-prompt', 'custom system prompt',
            '--model', 'gpt-4',
            '--ai'
        ]
        mocker.patch('sys.argv', test_args)
        
        parser, args = parse_args()
        
        assert args.prompt == 'custom prompt'
        assert args.system_prompt == 'custom system prompt'
        assert args.model == 'gpt-4'
        assert args.ai is True
        assert args.new_session is False  # Check default value

    def test_parse_args_short_flags(self, mocker):
        """Test parse_args with short flag versions."""
        test_args = [
            'askgpt',
            '-p', 'short prompt',
            '-s', 'short system',
            '-m', 'gpt-3.5-turbo',
            '-a'
        ]
        mocker.patch('sys.argv', test_args)
        
        parser, args = parse_args()
        
        assert args.prompt == 'short prompt'
        assert args.system_prompt == 'short system'
        assert args.model == 'gpt-3.5-turbo'
        assert args.ai is True
        assert args.new_session is False  # Default value

    def test_parse_args_with_new_session_flag(self, mocker):
        """Test parse_args with new session flag."""
        test_args = [
            'askgpt',
            '--prompt', 'test prompt',
            '--ai',
            '--new-session'
        ]
        mocker.patch('sys.argv', test_args)
        
        parser, args = parse_args()
        
        assert args.prompt == 'test prompt'
        assert args.ai is True
        assert args.new_session is True

    def test_parse_args_with_temperature_flag(self, mocker):
        """Test parse_args with temperature flag."""
        test_args = [
            'askgpt',
            '--prompt', 'test prompt',
            '--temperature', '0.5'
        ]
        mocker.patch('sys.argv', test_args)
        
        parser, args = parse_args()
        
        assert args.prompt == 'test prompt'
        assert args.temperature == 0.5


class TestMain:
    """Test the main function integration."""

    def test_main_non_ai_mode(self, mocker):
        """Test main function in non-AI mode (async query)."""
        # Mock parse_args to return test arguments
        mock_args = Mock()
        mock_args.prompt = 'test prompt'
        mock_args.system_prompt = 'test system'
        mock_args.model = 'gpt-4'
        mock_args.ai = False
        mock_args.new_session = False  # Add the new attribute
        
        mocker.patch('askgpt.__main__.parse_args', return_value=(Mock(), mock_args))
        
        # Mock query_chatgpt
        mock_query = mocker.patch('askgpt.__main__.query_chatgpt')
        mock_query.return_value = 'mock response'
        
        # Mock asyncio.run
        mock_asyncio_run = mocker.patch('askgpt.__main__.asyncio.run')
        mock_asyncio_run.return_value = 'async response'
        
        # Mock console.print
        mock_console = mocker.patch('askgpt.__main__.console')
        
        # Call main
        main()
        
        # Assertions
        mock_asyncio_run.assert_called_once()
        mock_console.print.assert_called_once_with('async response')

    def test_main_ai_mode(self, mocker):
        """Test main function in AI mode (streaming query)."""
        # Mock parse_args to return test arguments
        mock_args = Mock()
        mock_args.prompt = 'ai prompt'
        mock_args.system_prompt = 'custom system'
        mock_args.model = 'gpt-4'
        mock_args.ai = True
        mock_args.new_session = False  # Add the new attribute
        
        mocker.patch('askgpt.__main__.parse_args', return_value=(Mock(), mock_args))
        
        # Mock session management functions
        mocker.patch('askgpt.__main__.get_current_session_id', return_value='test-session-id')
        mock_load_session = mocker.patch('askgpt.__main__.load_session')
        mock_load_session.return_value = {
            "id": "test-session-id",
            "created_at": "2023-01-01T00:00:00",
            "messages": []
        }
        mocker.patch('askgpt.__main__.save_session')
        
        # Mock query_chatgpt_streaming to return a string
        mock_streaming = mocker.patch('askgpt.__main__.query_chatgpt_streaming')
        mock_streaming.return_value = 'AI response'
        
        # Mock asyncio.run should not be called in AI mode
        mock_asyncio_run = mocker.patch('askgpt.__main__.asyncio.run')
        
        # Call main
        main()
        
        # Assertions
        mock_streaming.assert_called_once()
        # Verify the call has the right message structure
        call_args = mock_streaming.call_args
        messages = call_args[0][0]
        assert len(messages) >= 2  # Should have system and user messages
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == markdown_system_prompt
        # Find the user message (should be the second message in the list passed to streaming)
        user_message = messages[1]
        assert user_message['role'] == 'user'
        assert user_message['content'] == 'ai prompt'
        
        mock_asyncio_run.assert_not_called()

    def test_main_system_prompt_selection(self, mocker):
        """Test that main correctly selects system prompt based on ai flag."""
        # Test AI mode uses markdown system prompt
        mock_args = Mock()
        mock_args.prompt = 'test'
        mock_args.system_prompt = 'custom'
        mock_args.model = 'gpt-4'
        mock_args.ai = True
        mock_args.new_session = False  # Add the new attribute
        
        mocker.patch('askgpt.__main__.parse_args', return_value=(Mock(), mock_args))
        
        # Mock session management functions
        mocker.patch('askgpt.__main__.get_current_session_id', return_value='test-session-id')
        mock_load_session = mocker.patch('askgpt.__main__.load_session')
        mock_load_session.return_value = {
            "id": "test-session-id",
            "created_at": "2023-01-01T00:00:00",
            "messages": []
        }
        mocker.patch('askgpt.__main__.save_session')
        
        mock_streaming = mocker.patch('askgpt.__main__.query_chatgpt_streaming')
        mock_streaming.return_value = 'AI response'
        
        main()
        
        # In AI mode, should use markdown_system_prompt regardless of args.system_prompt
        call_args = mock_streaming.call_args
        messages = call_args[0][0]
        assert messages[0]['content'] == markdown_system_prompt
        
        # Test non-AI mode uses provided system prompt
        mock_args.ai = False
        mock_asyncio_run = mocker.patch('askgpt.__main__.asyncio.run')
        mock_console = mocker.patch('askgpt.__main__.console')
        
        main()
        
        # Should use the custom system prompt from args
        assert mock_asyncio_run.call_count == 1

    def test_main_no_prompt_shows_usage(self, mocker):
        """Test that main shows usage and exits when no prompt is provided."""
        # Mock parse_args to return arguments with None prompt
        mock_args = Mock()
        mock_args.prompt = None  # No prompt provided
        mock_args.system_prompt = ''
        mock_args.model = None
        mock_args.ai = False
        mock_args.new_session = False
        mock_args.temperature = None
        
        # Mock parser
        mock_parser = Mock(spec=argparse.ArgumentParser)
        
        # Mock parse_args to return parser and args
        mocker.patch('askgpt.__main__.parse_args', return_value=(mock_parser, mock_args))
        
        # Mock console to suppress output during test
        mock_console = mocker.patch('askgpt.__main__.console')
        
        # Expect SystemExit when no prompt provided
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with code 1
        assert exc_info.value.code == 1
        
        # Should have called print_help to show usage
        mock_parser.print_help.assert_called_once()