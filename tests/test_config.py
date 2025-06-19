"""Tests for configuration functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from askgpt.__main__ import (
    get_config_dir,
    get_config_file,
    load_config,
    save_config,
    get_default_config,
)


class TestConfiguration:
    """Test configuration management functions."""

    def test_config_directory_creation(self, mocker):
        """Test that configuration directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            config_dir = get_config_dir()
            
            assert config_dir == mock_home / ".config" / "askgpt"
            assert config_dir.exists()

    def test_config_file_path(self, mocker):
        """Test that config file path is correct."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            config_file = get_config_file()
            
            assert config_file == mock_home / ".config" / "askgpt" / "askgpt.toml"

    def test_load_config_nonexistent_file(self, mocker):
        """Test loading config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            config = load_config()
            
            assert config == {}

    def test_save_and_load_config(self, mocker):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_home = Path(temp_dir)
            mocker.patch('askgpt.__main__.Path.home', return_value=mock_home)
            
            # Save test configuration
            test_config = {
                "model": "gpt-4",
                "temperature": 0.5
            }
            save_config(test_config)
            
            # Load and verify
            loaded_config = load_config()
            assert loaded_config == test_config

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

    def test_get_default_config(self):
        """Test that default config contains expected values."""
        defaults = get_default_config()
        
        assert "model" in defaults
        assert "temperature" in defaults
        assert defaults["model"] == "gpt-4o-mini"
        assert defaults["temperature"] == 0.7