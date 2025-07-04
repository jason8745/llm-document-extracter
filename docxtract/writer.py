"""
Markdown file writer for docxtract.

This module provides the MarkdownWriter class, which:
- Formats ExtractedDocument objects as Markdown
- Handles file writing and preview functionality

See .github/copilot-instructions.md for documentation and code style standards.
"""

from pathlib import Path
from typing import Optional

from .models import ExtractedDocument


class MarkdownWriter:
    """Handles writing extracted documents to Markdown format."""

    def __init__(self):
        """Initialize the Markdown writer."""
        pass

    def write_document(self, document: ExtractedDocument, output_path: Path) -> None:
        """
        Write an extracted document to a Markdown file.

        Args:
            document: ExtractedDocument to write
            output_path: Path where to save the Markdown file

        Raises:
            Exception: If writing fails
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            content = self._format_document(document)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            raise Exception(
                f"Failed to write Markdown file: {str(e)}. Please check the output path and permissions."
            )

    def _format_document(self, document: ExtractedDocument) -> str:
        """
        Format an ExtractedDocument as Markdown content.

        Args:
            document: Document to format

        Returns:
            Markdown-formatted string
        """
        lines = []

        # Add title if available
        if document.title:
            lines.append(f"# {document.title}")
            lines.append("")

        # Add Chinese summary if available
        if getattr(document, "summary_zh", None):
            lines.append("## 中文摘要")
            lines.append("")
            lines.append(document.summary_zh)
            lines.append("")

        # Add all sections
        for section in document.sections:
            # Skip "Full Content" section if we have parsed sections
            if section.title == "Full Content" and len(document.sections) > 1:
                continue

            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        # Add metadata footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Extracted from: {document.source_file}*")
        if getattr(document, "summary_zh", None):
            lines.append("*Chinese summary generated using GPT-4.1*")

        return "\n".join(lines)

    def preview_content(self, document: ExtractedDocument) -> str:
        """
        Generate a preview of the Markdown content without writing to file.

        Args:
            document: Document to preview

        Returns:
            Markdown-formatted preview string
        """
        return self._format_document(document)
