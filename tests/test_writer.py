"""Unit tests for docxtract.writer module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from docxtract.models import ExtractedDocument, DocumentSection
from docxtract.writer import MarkdownWriter


class TestMarkdownWriter:
    """Test cases for MarkdownWriter class."""

    @pytest.fixture
    def writer(self):
        """Create a MarkdownWriter instance."""
        return MarkdownWriter()

    @pytest.fixture
    def sample_document_with_summary(self):
        """Create a sample ExtractedDocument with Chinese summary."""
        sections = [
            DocumentSection(title="Abstract", content="This is the abstract content."),
            DocumentSection(title="Introduction", content="This is the introduction content."),
            DocumentSection(title="Method", content="This is the method content."),
        ]
        doc = ExtractedDocument(
            title="Sample Paper Title",
            sections=sections,
            source_file="sample.pdf"
        )
        # Add Chinese summary using setattr to simulate the dynamic field
        doc.summary_zh = "這是一篇關於機器學習的論文，主要探討了新的神經網絡架構。"
        return doc

    @pytest.fixture
    def sample_document_without_summary(self):
        """Create a sample ExtractedDocument without Chinese summary."""
        sections = [
            DocumentSection(title="Abstract", content="This is the abstract content."),
            DocumentSection(title="Results", content="This is the results content."),
        ]
        return ExtractedDocument(
            title="Another Paper",
            sections=sections,
            source_file="another.pdf"
        )

    @pytest.fixture
    def document_without_title(self):
        """Create a document without title."""
        sections = [
            DocumentSection(title="Full Content", content="This is the full document content."),
        ]
        return ExtractedDocument(
            title=None,
            sections=sections,
            source_file="untitled.pdf"
        )

    def test_init(self, writer):
        """Test MarkdownWriter initialization."""
        assert isinstance(writer, MarkdownWriter)

    def test_format_document_with_summary(self, writer, sample_document_with_summary):
        """Test formatting document with Chinese summary."""
        result = writer._format_document(sample_document_with_summary)
        
        # Check title
        assert "# Sample Paper Title" in result
        
        # Check Chinese summary section
        assert "## 中文摘要" in result
        assert "這是一篇關於機器學習的論文" in result
        
        # Check sections
        assert "## Abstract" in result
        assert "This is the abstract content." in result
        assert "## Introduction" in result
        assert "This is the introduction content." in result
        assert "## Method" in result
        assert "This is the method content." in result
        
        # Check metadata footer
        assert "---" in result
        assert "*Extracted from: sample.pdf*" in result
        assert "*Chinese summary generated using GPT-4.1*" in result

    def test_format_document_without_summary(self, writer, sample_document_without_summary):
        """Test formatting document without Chinese summary."""
        result = writer._format_document(sample_document_without_summary)
        
        # Check title
        assert "# Another Paper" in result
        
        # Check no Chinese summary section
        assert "## 中文摘要" not in result
        
        # Check sections
        assert "## Abstract" in result
        assert "This is the abstract content." in result
        assert "## Results" in result
        assert "This is the results content." in result
        
        # Check metadata footer
        assert "---" in result
        assert "*Extracted from: another.pdf*" in result
        # Should not mention GPT-4.1 when no summary
        assert "*Chinese summary generated using GPT-4.1*" not in result

    def test_format_document_without_title(self, writer, document_without_title):
        """Test formatting document without title."""
        result = writer._format_document(document_without_title)
        
        # Should not have title header when title is None
        lines = result.split('\n') if result else []
        first_non_empty_line = next((line for line in lines if line.strip()), "")
        assert not first_non_empty_line.startswith("# ")
        
        # Should include "Full Content" section when it's the only section
        assert "## Full Content" in result
        assert "This is the full document content." in result
        
        # Check metadata footer
        assert "*Extracted from: untitled.pdf*" in result

    def test_format_document_skips_full_content_with_multiple_sections(self, writer):
        """Test that Full Content section is skipped when other sections exist."""
        sections = [
            DocumentSection(title="Abstract", content="Abstract content"),
            DocumentSection(title="Full Content", content="Full document content"),
            DocumentSection(title="Conclusion", content="Conclusion content"),
        ]
        doc = ExtractedDocument(
            title="Test Paper",
            sections=sections,
            source_file="test.pdf"
        )
        
        result = writer._format_document(doc)
        
        # Should include other sections
        assert "## Abstract" in result
        assert "## Conclusion" in result
        
        # Should skip Full Content section
        assert "## Full Content" not in result
        assert "Full document content" not in result

    def test_preview_content(self, writer, sample_document_with_summary):
        """Test preview content generation."""
        preview = writer.preview_content(sample_document_with_summary)
        formatted = writer._format_document(sample_document_with_summary)
        
        # Preview should be identical to formatted content
        assert preview == formatted

    def test_write_document_success(self, writer, sample_document_with_summary):
        """Test successful document writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "output.md"
            
            # Should create directory and write file
            writer.write_document(sample_document_with_summary, output_path)
            
            # Check file was created
            assert output_path.exists()
            assert output_path.is_file()
            
            # Check content
            content = output_path.read_text(encoding="utf-8")
            assert "# Sample Paper Title" in content
            assert "## 中文摘要" in content
            assert "*Extracted from: sample.pdf*" in content

    def test_write_document_creates_directory(self, writer, sample_document_without_summary):
        """Test that write_document creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "deep" / "output.md"
            
            # Parent directory should not exist initially
            assert not output_path.parent.exists()
            
            writer.write_document(sample_document_without_summary, output_path)
            
            # Parent directory should be created
            assert output_path.parent.exists()
            assert output_path.exists()

    def test_write_document_handles_permission_error(self, writer, sample_document_with_summary):
        """Test write_document error handling for permission issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "readonly" / "output.md"
            
            # Create parent directory and make it read-only
            output_path.parent.mkdir()
            output_path.parent.chmod(0o444)  # Read-only
            
            try:
                with pytest.raises(Exception) as exc_info:
                    writer.write_document(sample_document_with_summary, output_path)
                
                # Should provide helpful error message
                assert "Failed to write Markdown file" in str(exc_info.value)
                assert "Please check the output path and permissions" in str(exc_info.value)
            finally:
                # Restore permissions for cleanup
                output_path.parent.chmod(0o755)

    @patch("builtins.open", side_effect=IOError("Mock IO error"))
    def test_write_document_handles_io_error(self, mock_open, writer, sample_document_with_summary):
        """Test write_document error handling for IO errors."""
        output_path = Path("test_output.md")
        
        with pytest.raises(Exception) as exc_info:
            writer.write_document(sample_document_with_summary, output_path)
        
        assert "Failed to write Markdown file" in str(exc_info.value)
        assert "Mock IO error" in str(exc_info.value)

    def test_format_document_empty_sections(self, writer):
        """Test formatting document with empty sections."""
        sections = [
            DocumentSection(title="Empty Section", content=""),
            DocumentSection(title="Another Section", content="Some content"),
        ]
        doc = ExtractedDocument(
            title="Test Document",
            sections=sections,
            source_file="test.pdf"
        )
        
        result = writer._format_document(doc)
        
        # Should include empty sections
        assert "## Empty Section" in result
        assert "## Another Section" in result
        assert "Some content" in result

    def test_format_document_special_characters(self, writer):
        """Test formatting document with special characters."""
        sections = [
            DocumentSection(title="Special Characters", content="Content with émojis 🚀 and 特殊字符"),
        ]
        doc = ExtractedDocument(
            title="Paper with Special Characters: 測試 & símbolos",
            sections=sections,
            source_file="special_chars.pdf"
        )
        
        result = writer._format_document(doc)
        
        # Should preserve special characters
        assert "Paper with Special Characters: 測試 & símbolos" in result
        assert "Content with émojis 🚀 and 特殊字符" in result
