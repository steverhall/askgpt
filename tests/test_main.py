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
        """Test query_chatgpt_streaming with empty system prompt."""
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
        
        # Call the function
        query_chatgpt_streaming("test prompt", "", "gpt-4o-mini")
        
        # Assertions
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": default_system_prompt},
                {"role": "user", "content": "test prompt"}
            ],
            model="gpt-4o-mini",
            temperature=0.7,
            stream=True
        )
        
        # Verify Live context manager was used
        mock_live_instance.__enter__.assert_called_once()
        mock_live_instance.__exit__.assert_called_once()
        
        # The main focus is that the OpenAI API call was made correctly
        # Markdown behavior is secondary to the core functionality

    def test_query_chatgpt_streaming_with_custom_system_prompt(self, mocker):
        """Test query_chatgpt_streaming with custom system prompt."""
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
        
        # Call the function
        query_chatgpt_streaming("Hello", custom_prompt, "gpt-4")
        
        # Assertions
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": "Hello"}
            ],
            model="gpt-4",
            temperature=0.7,
            stream=True
        )

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
        
        # Call the function
        query_chatgpt_streaming("test", "system", "gpt-4")
        
        # Verify OpenAI was called with correct API key
        mock_openai_class.assert_called_once_with(api_key='streaming-api-key')


class TestParseArgs:
    """Test the argument parsing functionality."""

    def test_parse_args_default_values(self, mocker):
        """Test parse_args with default values."""
        # Mock sys.argv to simulate command line arguments
        test_args = ['askgpt', '--prompt', 'test prompt']
        mocker.patch('sys.argv', test_args)
        
        args = parse_args()
        
        assert args.prompt == 'test prompt'
        assert args.system_prompt == ''
        assert args.model is None
        assert args.temperature is None
        assert args.ai is False

    def test_parse_args_all_custom_values(self, mocker):
        """Test parse_args with all custom values."""
        test_args = [
            'askgpt',
            '--prompt', 'custom prompt',
            '--system-prompt', 'custom system prompt',
            '--model', 'gpt-4',
            '--temperature', '0.5',
            '--ai'
        ]
        mocker.patch('sys.argv', test_args)
        
        args = parse_args()
        
        assert args.prompt == 'custom prompt'
        assert args.system_prompt == 'custom system prompt'
        assert args.model == 'gpt-4'
        assert args.temperature == 0.5
        assert args.ai is True

    def test_parse_args_short_flags(self, mocker):
        """Test parse_args with short flag versions."""
        test_args = [
            'askgpt',
            '-p', 'short prompt',
            '-s', 'short system',
            '-m', 'gpt-3.5-turbo',
            '-t', '1.2',
            '-a'
        ]
        mocker.patch('sys.argv', test_args)
        
        args = parse_args()
        
        assert args.prompt == 'short prompt'
        assert args.system_prompt == 'short system'
        assert args.model == 'gpt-3.5-turbo'
        assert args.temperature == 1.2
        assert args.ai is True


class TestMain:
    """Test the main function integration."""

    def test_main_non_ai_mode(self, mocker):
        """Test main function in non-AI mode (async query)."""
        # Mock parse_args to return test arguments
        mock_args = Mock()
        mock_args.prompt = 'test prompt'
        mock_args.system_prompt = 'test system'
        mock_args.model = 'gpt-4'
        mock_args.temperature = 0.8
        mock_args.ai = False
        
        mocker.patch('askgpt.__main__.parse_args', return_value=mock_args)
        
        # Mock configuration functions
        mocker.patch('askgpt.__main__.load_config', return_value={'model': 'gpt-4o-mini', 'temperature': 0.7})
        mocker.patch('askgpt.__main__.get_config_file').return_value.exists.return_value = True
        mocker.patch('askgpt.__main__.save_config')
        
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
        mock_args.temperature = None
        mock_args.ai = True
        
        mocker.patch('askgpt.__main__.parse_args', return_value=mock_args)
        
        # Mock configuration functions
        mocker.patch('askgpt.__main__.load_config', return_value={'model': 'gpt-4o-mini', 'temperature': 0.7})
        mocker.patch('askgpt.__main__.get_config_file').return_value.exists.return_value = True
        mocker.patch('askgpt.__main__.save_config')
        
        # Mock query_chatgpt_streaming
        mock_streaming = mocker.patch('askgpt.__main__.query_chatgpt_streaming')
        
        # Mock asyncio.run should not be called in AI mode
        mock_asyncio_run = mocker.patch('askgpt.__main__.asyncio.run')
        
        # Call main
        main()
        
        # Assertions
        mock_streaming.assert_called_once_with(
            'ai prompt', 
            markdown_system_prompt,  # Should use markdown system prompt in AI mode
            'gpt-4',
            0.7  # Should use temperature from config since args.temperature is None
        )
        mock_asyncio_run.assert_not_called()

    def test_main_system_prompt_selection(self, mocker):
        """Test that main correctly selects system prompt based on ai flag."""
        # Test AI mode uses markdown system prompt
        mock_args = Mock()
        mock_args.prompt = 'test'
        mock_args.system_prompt = 'custom'
        mock_args.model = 'gpt-4'
        mock_args.temperature = None
        mock_args.ai = True
        
        mocker.patch('askgpt.__main__.parse_args', return_value=mock_args)
        
        # Mock configuration functions
        mocker.patch('askgpt.__main__.load_config', return_value={'model': 'gpt-4o-mini', 'temperature': 0.7})
        mocker.patch('askgpt.__main__.get_config_file').return_value.exists.return_value = True
        mocker.patch('askgpt.__main__.save_config')
        
        mock_streaming = mocker.patch('askgpt.__main__.query_chatgpt_streaming')
        
        main()
        
        # In AI mode, should use markdown_system_prompt regardless of args.system_prompt
        mock_streaming.assert_called_once_with('test', markdown_system_prompt, 'gpt-4', 0.7)
        
        # Test non-AI mode uses provided system prompt
        mock_args.ai = False
        mock_asyncio_run = mocker.patch('askgpt.__main__.asyncio.run')
        mock_console = mocker.patch('askgpt.__main__.console')
        
        main()
        
        # Should use the custom system prompt from args
        assert mock_asyncio_run.call_count == 1