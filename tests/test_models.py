"""
Unit tests for docxtract.models module.

Tests the Pydantic data models used throughout the application:
- DocumentSection: Individual sections of documents
- ExtractedDocument: Complete document representation
- SummaryRequest: Request model for summary generation
"""

import pytest
from pydantic import ValidationError

from docxtract.models import DocumentSection, ExtractedDocument, SummaryRequest


class TestDocumentSection:
    """Test DocumentSection model."""

    def test_valid_document_section(self):
        """Test creating a valid DocumentSection."""
        section = DocumentSection(
            title="Abstract",
            content="This is the abstract content.",
            page_numbers=[1, 2],
        )

        assert section.title == "Abstract"
        assert section.content == "This is the abstract content."
        assert section.page_numbers == [1, 2]

    def test_document_section_without_page_numbers(self):
        """Test DocumentSection with optional page_numbers field."""
        section = DocumentSection(title="Introduction", content="Introduction content.")

        assert section.title == "Introduction"
        assert section.content == "Introduction content."
        assert section.page_numbers is None

    def test_document_section_empty_content(self):
        """Test DocumentSection with empty content."""
        section = DocumentSection(title="Empty Section", content="")

        assert section.title == "Empty Section"
        assert section.content == ""

    def test_document_section_missing_required_fields(self):
        """Test DocumentSection validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentSection(title="Missing Content")

        assert "content" in str(exc_info.value)

    def test_document_section_invalid_page_numbers(self):
        """Test DocumentSection with invalid page numbers."""
        # This should still work as we accept any list
        section = DocumentSection(title="Test", content="Test content", page_numbers=[])
        assert section.page_numbers == []


class TestExtractedDocument:
    """Test ExtractedDocument model."""

    def test_valid_extracted_document(self, sample_extracted_document):
        """Test creating a valid ExtractedDocument."""
        doc = sample_extracted_document

        assert doc.title == "A Novel Approach to Document Extraction"
        assert doc.summary_zh is not None
        assert len(doc.sections) == 5
        assert doc.source_file == "/path/to/test/document.pdf"

    def test_extracted_document_minimal(self):
        """Test ExtractedDocument with minimal required fields."""
        doc = ExtractedDocument(sections=[], source_file="/path/to/document.pdf")

        assert doc.title is None
        assert doc.summary_zh is None
        assert doc.sections == []
        assert doc.source_file == "/path/to/document.pdf"

    def test_get_section_exists(self, sample_extracted_document):
        """Test get_section method with existing section."""
        doc = sample_extracted_document
        abstract = doc.get_section("Abstract")

        assert abstract is not None
        assert abstract.title == "Abstract"
        assert "novel approach" in abstract.content

    def test_get_section_case_insensitive(self, sample_extracted_document):
        """Test get_section method is case insensitive."""
        doc = sample_extracted_document
        abstract = doc.get_section("ABSTRACT")

        assert abstract is not None
        assert abstract.title == "Abstract"

    def test_get_section_not_exists(self, sample_extracted_document):
        """Test get_section method with non-existing section."""
        doc = sample_extracted_document
        section = doc.get_section("Non-Existing Section")

        assert section is None

    def test_has_section_exists(self, sample_extracted_document):
        """Test has_section method with existing section."""
        doc = sample_extracted_document

        assert doc.has_section("Abstract") is True
        assert doc.has_section("INTRODUCTION") is True  # Case insensitive

    def test_has_section_not_exists(self, sample_extracted_document):
        """Test has_section method with non-existing section."""
        doc = sample_extracted_document

        assert doc.has_section("Non-Existing") is False

    def test_extracted_document_missing_source_file(self):
        """Test ExtractedDocument validation with missing source_file."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedDocument(sections=[])

        assert "source_file" in str(exc_info.value)


class TestSummaryRequest:
    """Test SummaryRequest model."""

    def test_valid_summary_request(self, sample_summary_request):
        """Test creating a valid SummaryRequest."""
        request = sample_summary_request

        assert request.document_content is not None
        assert request.title == "Test Document"
        assert len(request.focus_areas) == 3
        assert "methodology" in request.focus_areas

    def test_summary_request_minimal(self):
        """Test SummaryRequest with minimal required fields."""
        request = SummaryRequest(document_content="Sample content for testing.")

        assert request.document_content == "Sample content for testing."
        assert request.title is None
        # Check default focus areas
        assert len(request.focus_areas) == 3
        assert "main contributions" in request.focus_areas
        assert "methodology" in request.focus_areas
        assert "results" in request.focus_areas

    def test_summary_request_custom_focus_areas(self):
        """Test SummaryRequest with custom focus areas."""
        request = SummaryRequest(
            document_content="Test content",
            title="Test Title",
            focus_areas=["custom area 1", "custom area 2"],
        )

        assert request.focus_areas == ["custom area 1", "custom area 2"]

    def test_summary_request_empty_focus_areas(self):
        """Test SummaryRequest with empty focus areas."""
        request = SummaryRequest(document_content="Test content", focus_areas=[])

        assert request.focus_areas == []

    def test_summary_request_missing_content(self):
        """Test SummaryRequest validation with missing document_content."""
        with pytest.raises(ValidationError) as exc_info:
            SummaryRequest(title="Test Title")

        assert "document_content" in str(exc_info.value)

    def test_summary_request_empty_content(self):
        """Test SummaryRequest with empty document content."""
        request = SummaryRequest(document_content="")

        assert request.document_content == ""


class TestModelIntegration:
    """Test integration between different models."""

    def test_document_with_multiple_sections(self):
        """Test creating document with multiple sections."""
        sections = [
            DocumentSection(title="Section 1", content="Content 1"),
            DocumentSection(title="Section 2", content="Content 2"),
            DocumentSection(title="Section 3", content="Content 3"),
        ]

        doc = ExtractedDocument(sections=sections, source_file="/test/file.pdf")

        assert len(doc.sections) == 3
        assert doc.get_section("Section 2").content == "Content 2"
        assert doc.has_section("Section 1") is True
        assert doc.has_section("Section 4") is False

    def test_document_to_summary_request(self, sample_extracted_document):
        """Test converting document content to summary request."""
        doc = sample_extracted_document

        # Combine all section content
        all_content = " ".join([section.content for section in doc.sections])

        request = SummaryRequest(
            document_content=all_content,
            title=doc.title,
            focus_areas=["main findings", "methodology"],
        )

        assert request.title == doc.title
        assert "novel approach" in request.document_content
        assert len(request.focus_areas) == 2

    @pytest.mark.parametrize(
        "section_name,expected_found",
        [
            ("Abstract", True),
            ("Introduction", True),
            ("Method", True),
            ("Results", True),
            ("Conclusion", True),
            ("Discussion", False),
            ("References", False),
        ],
    )
    def test_section_existence_parameterized(
        self, sample_extracted_document, section_name, expected_found
    ):
        """Parameterized test for section existence."""
        doc = sample_extracted_document
        assert doc.has_section(section_name) == expected_found
