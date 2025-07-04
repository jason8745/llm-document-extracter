"""
Unit tests for docxtract.parser module.

Tests the SectionParser class functionality:
- Section boundary detection using regex patterns
- Content extraction and structuring
- Document parsing integration
"""

import pytest

from docxtract.models import DocumentSection, ExtractedDocument
from docxtract.parser import SectionParser


class TestSectionParser:
    """Test SectionParser class."""

    def test_init_default_patterns(self):
        """Test SectionParser initialization with default patterns."""
        parser = SectionParser()

        # Should have compiled patterns
        assert len(parser.compiled_patterns) > 0
        assert all(hasattr(pattern, "match") for pattern, _ in parser.compiled_patterns)

    def test_init_custom_patterns(self):
        """Test SectionParser initialization with custom patterns."""
        custom_patterns = [
            (r"^Summary\s*$", "Summary"),
            (r"^Analysis\s*$", "Analysis"),
        ]

        parser = SectionParser(section_patterns=custom_patterns)

        assert len(parser.compiled_patterns) == 2
        # Check that custom patterns are used
        pattern_names = [name for _, name in parser.compiled_patterns]
        assert "Summary" in pattern_names
        assert "Analysis" in pattern_names

    def test_detect_section_boundaries_basic(self):
        """Test detection of basic section boundaries."""
        text = """
        Title of the paper
        
        Abstract
        This is the abstract content.
        
        Introduction
        This is the introduction content.
        
        Method
        This describes the methodology.
        
        Results
        These are the results.
        
        Conclusion
        This is the conclusion.
        """

        parser = SectionParser()
        boundaries = parser.detect_section_boundaries(text)

        expected_sections = [
            "Abstract",
            "Introduction",
            "Method",
            "Results",
            "Conclusion",
        ]
        detected_sections = [name for name, _ in boundaries]

        assert detected_sections == expected_sections
        assert len(boundaries) == 5

    def test_detect_section_boundaries_case_insensitive(self):
        """Test that section detection is case insensitive."""
        text = """
        ABSTRACT
        Content here.
        
        introduction
        More content.
        
        MeThOd
        Method content.
        """

        parser = SectionParser()
        boundaries = parser.detect_section_boundaries(text)

        detected_sections = [name for name, _ in boundaries]
        assert "Abstract" in detected_sections
        assert "Introduction" in detected_sections
        assert "Method" in detected_sections

    def test_detect_section_boundaries_no_sections(self):
        """Test detection when no section headers are found."""
        text = """
        This is just regular text without any
        section headers that we can detect.
        It should return an empty list.
        """

        parser = SectionParser()
        boundaries = parser.detect_section_boundaries(text)

        assert boundaries == []

    def test_detect_section_boundaries_with_whitespace(self):
        """Test detection of sections with extra whitespace."""
        text = """
        Abstract  
        Content here.
        
          Introduction   
        More content.
        
        Method    
        Method content.
        """

        parser = SectionParser()
        boundaries = parser.detect_section_boundaries(text)

        detected_sections = [name for name, _ in boundaries]
        assert "Abstract" in detected_sections
        assert "Introduction" in detected_sections
        assert "Method" in detected_sections

    @pytest.mark.parametrize(
        "section_header,expected_name",
        [
            ("Abstract", "Abstract"),
            ("Introduction", "Introduction"),
            ("Method", "Method"),
            ("Methods", "Method"),
            ("Methodology", "Methodology"),
            ("Results", "Results"),
            ("Result", "Results"),
            ("Discussion", "Discussion"),
            ("Conclusion", "Conclusion"),
            ("Conclusions", "Conclusion"),
            ("References", "References"),
            ("Reference", "References"),
            ("Bibliography", "References"),
        ],
    )
    def test_section_pattern_matching(self, section_header, expected_name):
        """Test individual section pattern matching."""
        text = f"{section_header}\nSome content here."

        parser = SectionParser()
        boundaries = parser.detect_section_boundaries(text)

        assert len(boundaries) == 1
        assert boundaries[0][0] == expected_name

    def test_extract_sections_basic(self):
        """Test basic section extraction."""
        text = """
        Title of Paper
        
        Abstract
        This is the abstract content.
        It spans multiple lines.
        
        Introduction
        This is the introduction.
        With some more details.
        
        Method
        Methodology description here.
        """

        parser = SectionParser()
        sections = parser.extract_sections(text)

        assert len(sections) == 3

        # Check Abstract section
        abstract = next(s for s in sections if s.title == "Abstract")
        assert "abstract content" in abstract.content
        assert "multiple lines" in abstract.content

        # Check Introduction section
        introduction = next(s for s in sections if s.title == "Introduction")
        assert "introduction" in introduction.content
        assert "more details" in introduction.content

        # Check Method section
        method = next(s for s in sections if s.title == "Method")
        assert "Methodology description" in method.content

    def test_extract_sections_no_headers(self):
        """Test extraction when no section headers are found."""
        text = "This is just plain text without any section headers."

        parser = SectionParser()
        sections = parser.extract_sections(text)

        assert len(sections) == 1
        assert sections[0].title == "Full Content"
        assert sections[0].content == text

    def test_extract_sections_empty_sections(self):
        """Test extraction with empty sections."""
        text = """
        Abstract
        
        Introduction
        This has content.
        
        Method
        
        Results
        This also has content.
        """

        parser = SectionParser()
        sections = parser.extract_sections(text)

        # Should only get sections with content
        section_titles = [s.title for s in sections]
        assert "Introduction" in section_titles
        assert "Results" in section_titles
        # Abstract and Method should be filtered out (empty content)
        assert len([s for s in sections if s.title == "Abstract"]) == 0
        assert len([s for s in sections if s.title == "Method"]) == 0

    def test_extract_sections_last_section(self):
        """Test that the last section is properly extracted."""
        text = """
        Introduction
        Introduction content here.
        
        Method
        Method content here.
        
        Conclusion
        This is the final conclusion.
        It should be properly extracted.
        """

        parser = SectionParser()
        sections = parser.extract_sections(text)

        conclusion = next(s for s in sections if s.title == "Conclusion")
        assert "final conclusion" in conclusion.content
        assert "properly extracted" in conclusion.content

    def test_parse_document_with_sections(self, sample_extracted_document):
        """Test parsing a document that contains section headers."""
        # Create a document with section headers in the content
        text_with_sections = """
        Abstract
        This paper presents a novel approach to document extraction.
        
        Introduction
        Document processing has become increasingly important.
        
        Method
        We propose a hybrid approach combining PDF parsing and NLP.
        
        Results
        Our method achieves 95% accuracy on test documents.
        
        Conclusion
        The proposed method shows significant improvements.
        """

        # Create document with the sectioned content
        document = ExtractedDocument(
            title="Test Document",
            sections=[
                DocumentSection(title="Full Content", content=text_with_sections)
            ],
            source_file="/test/file.pdf",
        )

        parser = SectionParser()
        parsed_document = parser.parse_document(document)

        assert len(parsed_document.sections) == 5
        section_titles = [s.title for s in parsed_document.sections]
        expected_titles = [
            "Abstract",
            "Introduction",
            "Method",
            "Results",
            "Conclusion",
        ]
        assert section_titles == expected_titles

    def test_parse_document_no_sections(self):
        """Test parsing a document without section headers."""
        document = ExtractedDocument(
            title="Test Document",
            sections=[
                DocumentSection(
                    title="Full Content", content="Plain text without sections"
                )
            ],
            source_file="/test/file.pdf",
        )

        parser = SectionParser()
        parsed_document = parser.parse_document(document)

        assert len(parsed_document.sections) == 1
        assert parsed_document.sections[0].title == "Full Content"
        assert parsed_document.sections[0].content == "Plain text without sections"

    def test_parse_document_empty_sections(self):
        """Test parsing a document with no sections."""
        document = ExtractedDocument(
            title="Test Document", sections=[], source_file="/test/file.pdf"
        )

        parser = SectionParser()
        parsed_document = parser.parse_document(document)

        assert len(parsed_document.sections) == 0

    def test_parse_document_preserves_metadata(self):
        """Test that document parsing preserves title and metadata."""
        document = ExtractedDocument(
            title="Original Title",
            summary_zh="Original summary",
            sections=[
                DocumentSection(title="Full Content", content="Abstract\nContent here.")
            ],
            source_file="/original/path.pdf",
        )

        parser = SectionParser()
        parsed_document = parser.parse_document(document)

        # Metadata should be preserved
        assert parsed_document.title == "Original Title"
        assert parsed_document.summary_zh == "Original summary"
        assert parsed_document.source_file == "/original/path.pdf"

        # But sections should be parsed
        assert len(parsed_document.sections) == 1
        assert parsed_document.sections[0].title == "Abstract"


class TestSectionParserIntegration:
    """Integration tests for SectionParser."""

    def test_realistic_paper_parsing(self):
        """Test parsing a realistic academic paper structure."""
        paper_text = """
        Artificial Intelligence in Document Processing:
        A Comprehensive Review
        
        Authors: John Smith, Jane Doe
        
        Abstract
        This paper provides a comprehensive review of artificial intelligence
        applications in document processing. We examine current methodologies,
        identify key challenges, and propose future research directions.
        
        Keywords: artificial intelligence, document processing, machine learning
        
        Introduction
        Document processing has emerged as a critical application area for
        artificial intelligence technologies. With the exponential growth of
        digital documents, automated processing has become essential.
        
        The main contributions of this work are:
        1. A systematic review of AI techniques in document processing
        2. Analysis of current challenges and limitations
        3. Recommendations for future research
        
        Methodology
        We conducted a systematic literature review covering publications
        from 2015 to 2023. Our search strategy included multiple databases
        and used carefully constructed search terms.
        
        Results
        Our analysis identified 150 relevant studies across five major
        categories of AI applications in document processing.
        
        Key findings include:
        - 60% of studies focus on text extraction
        - 25% address document classification
        - 15% deal with structure recognition
        
        Discussion
        The results highlight significant progress in AI-based document
        processing, while also revealing important gaps in current research.
        
        Conclusion
        This review demonstrates the maturity of AI applications in document
        processing while identifying promising directions for future work.
        
        References
        [1] Smith, J. et al. (2020). Machine Learning for Documents.
        [2] Doe, J. et al. (2021). AI in Text Processing.
        """

        parser = SectionParser()
        sections = parser.extract_sections(paper_text)

        # Should extract all major sections
        section_titles = [s.title for s in sections]
        expected_sections = [
            "Abstract",
            "Introduction",
            "Methodology",
            "Results",
            "Discussion",
            "Conclusion",
            "References",
        ]

        assert all(title in section_titles for title in expected_sections)

        # Check content quality
        abstract = next(s for s in sections if s.title == "Abstract")
        assert "comprehensive review" in abstract.content
        assert "Keywords:" not in abstract.content  # Should not include keywords line

        introduction = next(s for s in sections if s.title == "Introduction")
        assert "main contributions" in introduction.content

        results = next(s for s in sections if s.title == "Results")
        assert "150 relevant studies" in results.content

    def test_custom_section_patterns(self):
        """Test using custom section patterns for different document types."""
        # Define patterns for a technical report
        custom_patterns = [
            (r"^Executive Summary\s*$", "Executive Summary"),
            (r"^Problem Statement\s*$", "Problem Statement"),
            (r"^Proposed Solution\s*$", "Proposed Solution"),
            (r"^Implementation\s*$", "Implementation"),
            (r"^Evaluation\s*$", "Evaluation"),
            (r"^Recommendations\s*$", "Recommendations"),
        ]

        report_text = """
        Technical Report: System Optimization
        
        Executive Summary
        This report presents findings from our system optimization project.
        
        Problem Statement
        Current system performance is below acceptable thresholds.
        
        Proposed Solution
        We recommend implementing a new caching strategy.
        
        Implementation
        The solution was implemented over three phases.
        
        Evaluation
        Testing shows 40% improvement in response times.
        
        Recommendations
        We recommend full deployment of the optimized system.
        """

        parser = SectionParser(section_patterns=custom_patterns)
        sections = parser.extract_sections(report_text)

        section_titles = [s.title for s in sections]
        expected_titles = [
            "Executive Summary",
            "Problem Statement",
            "Proposed Solution",
            "Implementation",
            "Evaluation",
            "Recommendations",
        ]

        assert section_titles == expected_titles

        # Verify content
        exec_summary = next(s for s in sections if s.title == "Executive Summary")
        assert "optimization project" in exec_summary.content
