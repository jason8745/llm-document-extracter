"""Shared utility functions for docxtract."""

import os
from pathlib import Path
from typing import Optional


def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def validate_pdf_file(file_path: Path) -> bool:
    """
    Validate that a file is a readable PDF.

    Args:
        file_path: Path to the file to validate

    Returns:
        True if file is a valid PDF, False otherwise
    """
    if not file_path.exists():
        return False

    if not file_path.is_file():
        return False

    if file_path.suffix.lower() != ".pdf":
        return False

    return True


def get_default_output_path(input_path: Path) -> Path:
    """
    Generate a default output path for a given input PDF.

    Args:
        input_path: Path to the input PDF file

    Returns:
        Default output path with .md extension
    """
    return input_path.with_suffix(".md")


def check_azure_config() -> bool:
    """
    Check if Azure OpenAI configuration is available.

    Returns:
        True if Azure OpenAI is configured, False otherwise
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    return bool(endpoint and api_key)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted file size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
