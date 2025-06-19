"""Integration test for AI session management."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from askgpt.__main__ import main, parse_args


class TestAISessionIntegration:
    """Test the complete AI session workflow."""

    def test_ai_mode_session_workflow(self, mocker):
        """Test complete workflow: new session -> conversation -> continuation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Mock OpenAI API calls
            mock_streaming = mocker.patch('askgpt.__main__.query_chatgpt_streaming')
            mock_streaming.return_value = 'Hello! How can I help you?'
            
            # Mock environment variable
            mocker.patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
            
            # Mock Live and Markdown for streaming display
            mock_live_instance = Mock()
            mocker.patch('askgpt.__main__.Live', return_value=mock_live_instance)
            mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
            mock_live_instance.__exit__ = Mock(return_value=None)
            mocker.patch('askgpt.__main__.Markdown')
            
            # Test 1: Start a new AI session
            test_args = ['askgpt', '--ai', '--new-session', '--prompt', 'Hello AI']
            mocker.patch('sys.argv', test_args)
            
            main()
            
            # Verify session was created and used
            config_dir = mock_home / ".config" / "askgpt"
            sessions_dir = config_dir / "sessions"
            
            assert config_dir.exists()
            assert sessions_dir.exists()
            
            # Should have a config file with current session
            config_file = config_dir / "askgpt.toml"
            assert config_file.exists()
            
            # Should have exactly one session file
            session_files = list(sessions_dir.glob("*.json"))
            assert len(session_files) == 1
            
            # Verify the streaming function was called with correct structure
            assert mock_streaming.call_count == 1
            call_args = mock_streaming.call_args[0]
            messages = call_args[0]
            
            # Should have system prompt and user message (before response was added)
            assert len(messages) >= 2
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'
            assert messages[1]['content'] == 'Hello AI'
            
            # Test 2: Continue the same session (without --new-session)
            mock_streaming.reset_mock()
            mock_streaming.return_value = 'Sure, I can help with that!'
            
            test_args = ['askgpt', '--ai', '--prompt', 'Can you help me?']
            mocker.patch('sys.argv', test_args)
            
            main()
            
            # Should still have only one session file
            session_files = list(sessions_dir.glob("*.json"))
            assert len(session_files) == 1
            
            # Verify the streaming function was called with conversation history
            assert mock_streaming.call_count == 1
            call_args = mock_streaming.call_args[0]
            messages = call_args[0]
            
            # Should now have: system + user1 + assistant1 + user2 (and possibly assistant2 added after)
            assert len(messages) >= 4
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'
            assert messages[1]['content'] == 'Hello AI'
            assert messages[2]['role'] == 'assistant'
            assert messages[2]['content'] == 'Hello! How can I help you?'
            assert messages[3]['role'] == 'user'
            assert messages[3]['content'] == 'Can you help me?'

    def test_new_session_flag_creates_fresh_session(self, mocker):
        """Test that --new-session creates a new session even when one exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Mock OpenAI API calls
            mock_streaming = mocker.patch('askgpt.__main__.query_chatgpt_streaming')
            mock_streaming.return_value = 'New session started!'
            
            # Mock environment variable
            mocker.patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
            
            # Mock Live and Markdown
            mock_live_instance = Mock()
            mocker.patch('askgpt.__main__.Live', return_value=mock_live_instance)
            mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
            mock_live_instance.__exit__ = Mock(return_value=None)
            mocker.patch('askgpt.__main__.Markdown')
            
            sessions_dir = mock_home / ".config" / "askgpt" / "sessions"
            
            # Create first session
            test_args = ['askgpt', '--ai', '--prompt', 'First session']
            mocker.patch('sys.argv', test_args)
            main()
            
            # Should have one session
            session_files = list(sessions_dir.glob("*.json"))
            assert len(session_files) == 1
            first_session_file = session_files[0]
            
            # Create new session with --new-session flag
            mock_streaming.reset_mock()
            test_args = ['askgpt', '--ai', '--new-session', '--prompt', 'Second session']
            mocker.patch('sys.argv', test_args)
            main()
            
            # Should now have two session files
            session_files = list(sessions_dir.glob("*.json"))
            assert len(session_files) == 2
            
            # New session should only have system prompt + new user message (before response)
            call_args = mock_streaming.call_args[0]
            messages = call_args[0]
            assert len(messages) >= 2
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'
            assert messages[1]['content'] == 'Second session'

    def test_non_ai_mode_unaffected(self, mocker):
        """Test that non-AI mode doesn't create sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Mock the async query function
            mock_query = mocker.patch('askgpt.__main__.query_chatgpt')
            mock_asyncio_run = mocker.patch('askgpt.__main__.asyncio.run')
            mock_asyncio_run.return_value = 'ls -la'
            mock_console = mocker.patch('askgpt.__main__.console')
            
            # Mock environment variable
            mocker.patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
            
            # Test non-AI mode
            test_args = ['askgpt', '--prompt', 'list files']
            mocker.patch('sys.argv', test_args)
            
            main()
            
            # Should not create any session directories
            config_dir = mock_home / ".config" / "askgpt"
            if config_dir.exists():
                sessions_dir = config_dir / "sessions"
                if sessions_dir.exists():
                    session_files = list(sessions_dir.glob("*.json"))
                    assert len(session_files) == 0
            
            # Should have called the non-AI async function
            assert mock_asyncio_run.call_count == 1
            assert mock_console.print.call_count == 1