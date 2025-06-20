"""Tests for session management functionality."""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from askgpt.__main__ import (
    create_new_session,
    get_config_dir,
    get_current_session_id,
    get_sessions_dir,
    load_config,
    load_session,
    save_config,
    save_session,
    set_current_session_id,
)


class TestSessionManagement:
    """Test session management functions."""

    def test_config_directory_creation(self, mocker):
        """Test that configuration directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            config_dir = get_config_dir()
            
            assert config_dir == mock_home / ".config" / "askgpt"
            assert config_dir.exists()

    def test_sessions_directory_creation(self, mocker):
        """Test that sessions directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            sessions_dir = get_sessions_dir()
            
            assert sessions_dir == mock_home / ".config" / "askgpt" / "sessions"
            assert sessions_dir.exists()

    def test_create_new_session(self, mocker):
        """Test creating a new session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Mock uuid to get a predictable session ID
            test_uuid = "test-session-id"
            mocker.patch('askgpt.__main__.uuid.uuid4', return_value=test_uuid)
            
            session_id = create_new_session()
            
            assert session_id == test_uuid
            
            # Check that session file was created
            session_file = get_sessions_dir() / f"{test_uuid}.json"
            assert session_file.exists()
            
            # Check session file content
            with open(session_file, "r") as f:
                session_data = json.load(f)
            
            assert session_data["id"] == test_uuid
            assert "created_at" in session_data
            assert session_data["messages"] == []

    def test_load_and_save_session(self, mocker):
        """Test loading and saving session data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Create a session
            session_id = create_new_session()
            
            # Load the session
            session_data = load_session(session_id)
            assert session_data["id"] == session_id
            assert session_data["messages"] == []
            
            # Add some messages and save
            session_data["messages"].append({"role": "user", "content": "Hello"})
            session_data["messages"].append({"role": "assistant", "content": "Hi there!"})
            save_session(session_data)
            
            # Load again and verify
            reloaded_data = load_session(session_id)
            assert len(reloaded_data["messages"]) == 2
            assert reloaded_data["messages"][0]["content"] == "Hello"
            assert reloaded_data["messages"][1]["content"] == "Hi there!"

    def test_load_nonexistent_session(self, mocker):
        """Test loading a nonexistent session returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            result = load_session("nonexistent-session")
            assert result is None

    def test_config_persistence(self, mocker):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Initially should have no current session
            assert get_current_session_id() is None
            
            # Set a current session
            test_session_id = "test-session-123"
            set_current_session_id(test_session_id)
            
            # Should be able to retrieve it
            assert get_current_session_id() == test_session_id
            
            # Should persist across loads
            config = load_config()
            assert config["current_session"] == test_session_id

    def test_config_file_corruption_handling(self, mocker):
        """Test handling of corrupted config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Create config directory
            config_dir = get_config_dir()
            
            # Create a corrupted config file
            config_file = config_dir / "askgpt.toml"
            with open(config_file, "w") as f:
                f.write("invalid toml content [[[")
            
            # Should handle corruption gracefully
            config = load_config()
            assert config == {}
            
            # Should be able to set and retrieve session after corruption
            set_current_session_id("new-session")
            assert get_current_session_id() == "new-session"

    def test_config_model_and_temperature_defaults(self, mocker):
        """Test that config can store and retrieve model and temperature settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Initially should have no config values
            config = load_config()
            assert config.get("model") is None
            assert config.get("temperature") is None
            
            # Set model and temperature in config
            config["model"] = "gpt-4"
            config["temperature"] = 0.5
            save_config(config)
            
            # Should be able to retrieve them
            loaded_config = load_config()
            assert loaded_config["model"] == "gpt-4"
            assert loaded_config["temperature"] == 0.5