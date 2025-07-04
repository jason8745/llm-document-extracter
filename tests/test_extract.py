"""
Unit tests for docxtract.extract module.

Tests the PDFExtractor class functionality:
- Text extraction from PDF files
- Title detection using heuristics
- Document structure creation
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docxtract.extract import PDFExtractor
from docxtract.models import ExtractedDocument


class TestPDFExtractor:
    """Test PDFExtractor class."""

    def test_init(self):
        """Test PDFExtractor initialization."""
        extractor = PDFExtractor()
        assert extractor is not None

    @patch('docxtract.extract.fitz')
    def test_extract_text_from_pdf_success(self, mock_fitz):
        """Test successful text extraction from PDF."""
        # Mock PDF document and pages
        mock_doc = Mock()
        mock_doc.page_count = 2
        
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Page 1 content"
        
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Page 2 content"
        
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_fitz.open.return_value = mock_doc
        
        extractor = PDFExtractor()
        
        # Create a temporary file path
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)
        
        try:
            result = extractor.extract_text_from_pdf(pdf_path)
            
            # Verify the result
            expected_text = "Page 1 content\nPage 2 content"
            assert result == expected_text
            
            # Verify mocks were called correctly
            mock_fitz.open.assert_called_once_with(pdf_path)
            assert mock_doc.load_page.call_count == 2
            mock_doc.close.assert_called_once()
            
        finally:
            # Clean up
            pdf_path.unlink(missing_ok=True)

    def test_extract_text_from_pdf_file_not_found(self):
        """Test text extraction with non-existent file."""
        extractor = PDFExtractor()
        non_existent_path = Path("/path/to/non/existent/file.pdf")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            extractor.extract_text_from_pdf(non_existent_path)
        
        assert "PDF file not found" in str(exc_info.value)

    @patch('docxtract.extract.fitz')
    def test_extract_text_from_pdf_extraction_error(self, mock_fitz):
        """Test text extraction with PDF reading error."""
        mock_fitz.open.side_effect = Exception("Corrupted PDF")
        
        extractor = PDFExtractor()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)
        
        try:
            with pytest.raises(Exception) as exc_info:
                extractor.extract_text_from_pdf(pdf_path)
            
            assert "Failed to extract text from PDF" in str(exc_info.value)
            assert "corrupted" in str(exc_info.value).lower()
            
        finally:
            pdf_path.unlink(missing_ok=True)

    @pytest.mark.parametrize("text,expected_title", [ 
        ( 
            "Advanced Machine Learning Techniques\nfor Document Processing\n\nAbstract\nThis paper presents...", 
            "Advanced Machine Learning Techniques for Document Processing" 
        ), 
        ( 
            "A Novel Approach to PDF Extraction\n\nJohn Doe\nUniversity of Example\n\nAbstract\nWe propose...", 
            "A Novel Approach to PDF Extraction" 
        ), 
    ])
    def test_detect_title(self, text, expected_title):
        """Test title detection with various text formats."""
        extractor = PDFExtractor()
        result = extractor.detect_title(text)
        assert result == expected_title

    def test_detect_title_with_author_info(self):
        """Test title detection stops at author information."""
        text = """
        Innovative Document Processing Framework
        
        Authors: Jane Smith, Bob Johnson
        Research Institute for AI
        
        Abstract
        This paper presents...
        """
        
        extractor = PDFExtractor()
        result = extractor.detect_title(text)
        assert result == "Innovative Document Processing Framework"

    def test_detect_title_with_keywords(self):
        """Test title detection stops at keywords section."""
        text = """
        Machine Learning for Text Analysis
        
        Keywords
        machine learning, text analysis, NLP
        
        Abstract
        This study...
        """
        
        extractor = PDFExtractor()
        result = extractor.detect_title(text)
        assert result == "Machine Learning for Text Analysis"

    def test_detect_title_multiline(self):
        """Test title detection with multi-line titles."""
        text = """
        Advanced Techniques in Natural Language Processing:
        A Comprehensive Study of Modern Approaches
        
        Abstract
        Recent advances...
        """
        
        extractor = PDFExtractor()
        result = extractor.detect_title(text)
        expected = "Advanced Techniques in Natural Language Processing: A Comprehensive Study of Modern Approaches"
        assert result == expected

    def test_detect_title_empty_text(self):
        """Test title detection with empty text."""
        extractor = PDFExtractor()
        result = extractor.detect_title("")
        assert result is None

    def test_detect_title_only_whitespace(self):
        """Test title detection with only whitespace."""
        extractor = PDFExtractor()
        result = extractor.detect_title("   \n\n   \n   ")
        assert result is None

    @patch.object(PDFExtractor, 'extract_text_from_pdf')
    @patch.object(PDFExtractor, 'detect_title')
    def test_extract_document_success(self, mock_detect_title, mock_extract_text):
        """Test successful document extraction."""
        # Setup mocks
        mock_text = "Sample PDF content for testing"
        mock_title = "Sample Document Title"
        
        mock_extract_text.return_value = mock_text
        mock_detect_title.return_value = mock_title
        
        extractor = PDFExtractor()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)
        
        try:
            result = extractor.extract_document(pdf_path)
            
            # Verify result type and structure
            assert isinstance(result, ExtractedDocument)
            assert result.title == mock_title
            assert result.source_file == str(pdf_path)
            assert len(result.sections) == 1
            assert result.sections[0].title == "Full Content"
            assert result.sections[0].content == mock_text
            
            # Verify mocks were called
            mock_extract_text.assert_called_once_with(pdf_path)
            mock_detect_title.assert_called_once_with(mock_text)
            
        finally:
            pdf_path.unlink(missing_ok=True)

    @patch.object(PDFExtractor, 'extract_text_from_pdf')
    @patch.object(PDFExtractor, 'detect_title')
    def test_extract_document_no_title(self, mock_detect_title, mock_extract_text):
        """Test document extraction when no title is detected."""
        mock_text = "Sample content without clear title"
        mock_extract_text.return_value = mock_text
        mock_detect_title.return_value = None
        
        extractor = PDFExtractor()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)
        
        try:
            result = extractor.extract_document(pdf_path)
            
            assert result.title is None
            assert result.sections[0].content == mock_text
            
        finally:
            pdf_path.unlink(missing_ok=True)

    @patch.object(PDFExtractor, 'extract_text_from_pdf')
    def test_extract_document_extraction_error(self, mock_extract_text):
        """Test document extraction when PDF extraction fails."""
        mock_extract_text.side_effect = Exception("PDF extraction failed")
        
        extractor = PDFExtractor()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)
        
        try:
            with pytest.raises(Exception) as exc_info:
                extractor.extract_document(pdf_path)
            
            assert "PDF extraction failed" in str(exc_info.value)
            
        finally:
            pdf_path.unlink(missing_ok=True)


class TestPDFExtractorIntegration:
    """Integration tests for PDFExtractor."""

    def test_title_detection_comprehensive(self):
        """Comprehensive test of title detection with realistic document."""
        text = """
        
        Artificial Intelligence in Document Processing:
        Advances and Future Directions
        
        John Smith¹, Jane Doe², Robert Johnson³
        
        ¹Department of Computer Science, University of Technology
        ²AI Research Lab, Tech Institute
        ³Digital Innovation Center
        
        email: john.smith@university.edu
        
        Abstract
        
        This paper presents a comprehensive review of artificial intelligence
        applications in document processing...
        
        Keywords: artificial intelligence, document processing, machine learning
        
        1. Introduction
        
        Document processing has become...
        """
        
        extractor = PDFExtractor()
        title = extractor.detect_title(text)
        
        expected = "Artificial Intelligence in Document Processing: Advances and Future Directions"
        assert title == expected

    def test_title_detection_edge_cases(self):
        """Test title detection with various edge cases."""
        extractor = PDFExtractor()
        
        # Case 1: Title with colon
        text1 = "Study Results: A Comprehensive Analysis\n\nAbstract\nContent..."
        assert extractor.detect_title(text1) == "Study Results: A Comprehensive Analysis"
        
        # Case 2: Very short lines that shouldn't be titles
        text2 = "A\nB\nC\nAbstract\nContent..."
        # Should not detect very short fragments as title
        result2 = extractor.detect_title(text2)
        assert result2 is None or len(result2) > 5
        
        # Case 3: Title followed immediately by abstract
        text3 = "Machine Learning Applications\nAbstract\nThis paper..."
        assert extractor.detect_title(text3) == "Machine Learning Applications"
