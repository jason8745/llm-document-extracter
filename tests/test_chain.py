"""Unit tests for docxtract.chain module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docxtract.chain import (
    DocExtractPipeline,
    extract_pdf_step,
    generate_application_ideas,
    generate_important_points,
    get_ideas_prompt,
    get_take_away_prompt,
    important_points_and_ideas_step,
    load_prompt_template,
    parse_sections_step,
    summarize_overall_step,
    summarize_sections_step,
    to_markdown_step,
)
from docxtract.models import DocumentSection, ExtractedDocument


class TestChainSteps:
    """Test cases for individual pipeline steps."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        sections = [
            DocumentSection(
                title="Abstract", content="This paper presents a novel approach."
            ),
            DocumentSection(
                title="Introduction", content="Background information here."
            ),
            DocumentSection(title="Method", content="Our methodology is as follows."),
        ]
        return ExtractedDocument(
            title="Test Paper", sections=sections, source_file="test.pdf"
        )

    def test_extract_pdf_step_success(self, sample_document):
        """Test successful PDF extraction step."""
        with patch("docxtract.chain.PDFExtractor") as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor.extract_document.return_value = sample_document
            mock_extractor_class.return_value = mock_extractor

            result = extract_pdf_step(Path("test.pdf"))

            assert "document" in result
            assert result["document"] == sample_document
            mock_extractor.extract_document.assert_called_once_with(Path("test.pdf"))

    def test_extract_pdf_step_failure(self):
        """Test PDF extraction step with extraction failure."""
        with patch("docxtract.chain.PDFExtractor") as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor.extract_document.side_effect = Exception(
                "PDF extraction failed"
            )
            mock_extractor_class.return_value = mock_extractor

            with pytest.raises(RuntimeError) as exc_info:
                extract_pdf_step(Path("test.pdf"))

            assert "Failed to extract PDF" in str(exc_info.value)
            assert "PDF extraction failed" in str(exc_info.value)

    def test_parse_sections_step_success(self, sample_document):
        """Test successful section parsing step."""
        with patch("docxtract.chain.SectionParser") as mock_parser_class:
            mock_parser = Mock()
            parsed_document = ExtractedDocument(
                title="Parsed Paper",
                sections=[
                    DocumentSection(title="Parsed Section", content="Parsed content")
                ],
                source_file="test.pdf",
            )
            mock_parser.parse_document.return_value = parsed_document
            mock_parser_class.return_value = mock_parser

            inputs = {"document": sample_document}
            result = parse_sections_step(inputs)

            assert "document" in result
            assert result["document"] == parsed_document
            mock_parser.parse_document.assert_called_once_with(sample_document)

    def test_parse_sections_step_failure(self, sample_document):
        """Test section parsing step with parsing failure."""
        with patch("docxtract.chain.SectionParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_document.side_effect = Exception("Parsing failed")
            mock_parser_class.return_value = mock_parser

            inputs = {"document": sample_document}

            with pytest.raises(RuntimeError) as exc_info:
                parse_sections_step(inputs)

            assert "Failed to parse sections" in str(exc_info.value)
            assert "Parsing failed" in str(exc_info.value)

    def test_summarize_sections_step_success(self, sample_document):
        """Test successful section summarization step."""
        with patch("docxtract.chain.ChineseSummarizer") as mock_summarizer_class:
            mock_summarizer = Mock()
            section_summaries = {
                "Abstract": "Abstract summary in Chinese",
                "Introduction": "Introduction summary in Chinese",
            }
            mock_summarizer.summarize_all_sections.return_value = section_summaries
            mock_summarizer_class.return_value = mock_summarizer

            inputs = {"document": sample_document}
            result = summarize_sections_step(inputs)

            assert "document" in result
            assert "section_summaries" in result
            assert result["document"] == sample_document
            assert result["section_summaries"] == section_summaries
            mock_summarizer.summarize_all_sections.assert_called_once_with(
                sample_document
            )

    def test_summarize_sections_step_failure(self, sample_document):
        """Test section summarization step with failure."""
        with patch("docxtract.chain.ChineseSummarizer") as mock_summarizer_class:
            mock_summarizer = Mock()
            mock_summarizer.summarize_all_sections.side_effect = Exception(
                "Summarization failed"
            )
            mock_summarizer_class.return_value = mock_summarizer

            inputs = {"document": sample_document}

            with pytest.raises(RuntimeError) as exc_info:
                summarize_sections_step(inputs)

            assert "Failed to summarize sections" in str(exc_info.value)
            assert "Summarization failed" in str(exc_info.value)

    def test_summarize_overall_step_success(self, sample_document):
        """Test successful overall summarization step."""
        with patch("docxtract.chain.ChineseSummarizer") as mock_summarizer_class:
            mock_summarizer = Mock()
            overall_summary = "Overall summary in Chinese"
            mock_summarizer.summarize_overall.return_value = overall_summary
            mock_summarizer_class.return_value = mock_summarizer

            inputs = {
                "document": sample_document,
                "section_summaries": {"Abstract": "Abstract summary"},
            }
            result = summarize_overall_step(inputs)

            assert "document" in result
            assert "section_summaries" in result
            assert "overall_summary" in result
            assert result["overall_summary"] == overall_summary
            mock_summarizer.summarize_overall.assert_called_once_with(
                {"Abstract": "Abstract summary"}, "Test Paper"
            )

    def test_summarize_overall_step_failure(self, sample_document):
        """Test overall summarization step with failure (should not raise)."""
        with patch("docxtract.chain.ChineseSummarizer") as mock_summarizer_class:
            mock_summarizer = Mock()
            mock_summarizer.summarize_overall.side_effect = Exception(
                "Overall summarization failed"
            )
            mock_summarizer_class.return_value = mock_summarizer

            inputs = {
                "document": sample_document,
                "section_summaries": {"Abstract": "Abstract summary"},
            }
            result = summarize_overall_step(inputs)

            # Should not raise exception, but return fallback message
            assert "overall_summary" in result
            assert result["overall_summary"] == "(Failed to generate overall summary)"

    def test_important_points_and_ideas_step_success(self, sample_document):
        """Test successful important points and ideas generation."""
        with (
            patch("docxtract.chain.generate_important_points") as mock_points,
            patch("docxtract.chain.generate_application_ideas") as mock_ideas,
        ):
            mock_points.return_value = "1. Important point one\n2. Important point two"
            mock_ideas.return_value = "1. Application idea one\n2. Application idea two"

            inputs = {
                "document": sample_document,
                "section_summaries": {"Abstract": "Summary"},
                "overall_summary": "Overall summary text",
            }
            result = important_points_and_ideas_step(inputs)

            assert "important_points" in result
            assert "ideas" in result
            assert (
                result["important_points"]
                == "1. Important point one\n2. Important point two"
            )
            assert result["ideas"] == "1. Application idea one\n2. Application idea two"

            # Verify the functions were called with correct arguments
            mock_points.assert_called_once()
            mock_ideas.assert_called_once()
            # The summarizer should be passed as first argument
            points_call_args = mock_points.call_args[0]
            ideas_call_args = mock_ideas.call_args[0]
            assert (
                points_call_args[1] == "Overall summary text"
            )  # summary is second arg
            assert ideas_call_args[1] == "Overall summary text"

    def test_important_points_and_ideas_step_with_real_functions(self, sample_document):
        """Test important points and ideas step with real function calls (integration-style)."""
        with patch("docxtract.chain.ChineseSummarizer") as mock_summarizer_class:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "   Response with whitespace   "
            mock_llm.invoke.return_value = mock_response

            mock_summarizer = Mock()
            mock_summarizer.llm = mock_llm
            mock_summarizer_class.return_value = mock_summarizer

            inputs = {
                "document": sample_document,
                "section_summaries": {"Abstract": "Summary"},
                "overall_summary": "Overall summary text",
            }
            result = important_points_and_ideas_step(inputs)

            assert "important_points" in result
            assert "ideas" in result
            # Should strip whitespace from responses
            assert result["important_points"] == "Response with whitespace"
            assert result["ideas"] == "Response with whitespace"
            assert mock_llm.invoke.call_count == 2  # Called for both points and ideas

    def test_important_points_and_ideas_step_no_llm(self, sample_document):
        """Test important points and ideas step when LLM is None."""
        with (
            patch("docxtract.chain.generate_important_points") as mock_points,
            patch("docxtract.chain.generate_application_ideas") as mock_ideas,
        ):
            mock_points.return_value = ""
            mock_ideas.return_value = ""

            inputs = {
                "document": sample_document,
                "section_summaries": {"Abstract": "Summary"},
                "overall_summary": "Overall summary text",
            }
            result = important_points_and_ideas_step(inputs)

            assert "important_points" in result
            assert "ideas" in result
            assert result["important_points"] == ""
            assert result["ideas"] == ""

    def test_important_points_and_ideas_step_failure(self, sample_document):
        """Test important points and ideas step with function failures."""
        with (
            patch("docxtract.chain.generate_important_points") as mock_points,
            patch("docxtract.chain.generate_application_ideas") as mock_ideas,
        ):
            mock_points.return_value = "(Failed to generate important points)"
            mock_ideas.return_value = "(Failed to generate application ideas)"

            inputs = {
                "document": sample_document,
                "section_summaries": {"Abstract": "Summary"},
                "overall_summary": "Overall summary text",
            }
            result = important_points_and_ideas_step(inputs)

            # Should not raise exception, but return fallback messages
            assert result["important_points"] == "(Failed to generate important points)"
            assert result["ideas"] == "(Failed to generate application ideas)"

    def test_to_markdown_step_complete(self, sample_document):
        """Test markdown generation with complete inputs."""
        inputs = {
            "document": sample_document,
            "section_summaries": {"Abstract": "Summary"},
            "overall_summary": "Complete overall summary",
            "important_points": "1. Point one\n2. Point two",
            "ideas": "1. Idea one\n2. Idea two",
        }

        result = to_markdown_step(inputs)

        # Check that all sections are included
        assert "# Test Paper" in result
        assert "## Top-5 Important Points" in result
        assert "1. Point one" in result
        assert "## Application Ideas" in result
        assert "1. Idea one" in result
        assert "## Chinese Summary" in result
        assert "Complete overall summary" in result
        assert "*Chinese summary generated using GPT-4.1*" in result

    def test_to_markdown_step_minimal(self, sample_document):
        """Test markdown generation with minimal inputs."""
        sample_document.title = None  # No title
        inputs = {
            "document": sample_document,
            "section_summaries": {},
            "overall_summary": None,  # No overall summary
        }

        result = to_markdown_step(inputs)

        # 應該只檢查是否有 H1 標題（# ），而不是 # 號開頭的所有行
        lines = result.splitlines()
        assert not any(
            line.startswith("# ") and not line.startswith("##") for line in lines
        )
        assert "## Top-5 Important Points" not in result  # No points section
        assert "## Application Ideas" not in result  # No ideas section
        assert "## Chinese Summary" in result
        assert "(Failed to generate overall summary)" in result
        assert "*Chinese summary generated using GPT-4.1*" in result

    def test_to_markdown_step_partial(self, sample_document):
        """Test markdown generation with partial inputs."""
        inputs = {
            "document": sample_document,
            "section_summaries": {"Abstract": "Summary"},
            "overall_summary": "Partial summary",
            "important_points": "Some points",
            # Missing "ideas"
        }

        result = to_markdown_step(inputs)

        # Should include what's available
        assert "# Test Paper" in result
        assert "## Top-5 Important Points" in result
        assert "Some points" in result
        assert "## Application Ideas" not in result  # Missing ideas
        assert "## Chinese Summary" in result
        assert "Partial summary" in result


class TestDocExtractPipeline:
    """Test cases for the complete pipeline."""

    def test_pipeline_is_runnable(self):
        """Test that DocExtractPipeline is properly constructed."""
        from langchain_core.runnables import Runnable

        # Should be a Runnable
        assert isinstance(DocExtractPipeline, Runnable)

    @pytest.mark.integration
    def test_pipeline_step_order(self):
        """Test that pipeline steps are in correct order."""
        # This is more of a smoke test to ensure the pipeline is constructible
        # Full integration testing would require real PDF files and LLM access

        # The pipeline should be constructible without errors
        assert DocExtractPipeline is not None

        # Pipeline should be composed of the expected steps
        # (This is a basic check - full integration would test execution)
        assert hasattr(DocExtractPipeline, "invoke")


class TestPromptTemplates:
    """Test cases for prompt templates and prompt loading functions."""

    def test_load_prompt_template_success(self, tmp_path):
        """Test successful loading of prompt template from file."""
        # Create a temporary template file
        template_file = tmp_path / "test_template.md"
        template_content = "Test template with {placeholder}"
        template_file.write_text(template_content, encoding="utf-8")

        result = load_prompt_template(template_file)

        assert result == template_content

    def test_load_prompt_template_file_not_found(self):
        """Test prompt template loading with non-existent file."""
        non_existent_path = Path("non_existent_template.md")

        with pytest.raises(FileNotFoundError):
            load_prompt_template(non_existent_path)

    def test_get_take_away_prompt_formatting(self):
        """Test take away prompt loading and formatting."""
        test_summary = "這是一個測試總結"

        result = get_take_away_prompt(test_summary)

        # Should contain the formatted summary
        assert test_summary in result
        # Should contain Chinese text indicators from the template
        assert "重點摘要提取" in result

    def test_get_ideas_prompt_formatting(self):
        """Test ideas prompt loading and formatting."""
        test_summary = "這是一個測試總結"

        result = get_ideas_prompt(test_summary)

        # Should contain the formatted summary
        assert test_summary in result
        # Should contain Chinese text indicators from the template
        assert "延伸應用" in result or "研究方向" in result

    @patch("docxtract.chain.load_prompt_template")
    def test_get_take_away_prompt_with_mock_template(self, mock_load):
        """Test take away prompt with mocked template loading."""
        mock_template = "Mock template: {summary}"
        mock_load.return_value = mock_template
        test_summary = "test summary"

        result = get_take_away_prompt(test_summary)

        assert result == "Mock template: test summary"
        mock_load.assert_called_once()

    @patch("docxtract.chain.load_prompt_template")
    def test_get_ideas_prompt_with_mock_template(self, mock_load):
        """Test ideas prompt with mocked template loading."""
        mock_template = "Mock ideas template: {summary}"
        mock_load.return_value = mock_template
        test_summary = "test summary"

        result = get_ideas_prompt(test_summary)

        assert result == "Mock ideas template: test summary"
        mock_load.assert_called_once()


class TestNewPromptFunctions:
    """Test cases for new prompt generation functions."""

    def test_generate_important_points_success(self):
        """Test successful generation of important points."""
        mock_summarizer = Mock()
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "1. Important point one\n2. Important point two"
        mock_llm.invoke.return_value = mock_response
        mock_summarizer.llm = mock_llm

        result = generate_important_points(mock_summarizer, "test summary")

        assert result == "1. Important point one\n2. Important point two"
        mock_llm.invoke.assert_called_once()
        # Verify the call contains a HumanMessage with formatted prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert "test summary" in call_args[0].content

    def test_generate_important_points_no_llm(self):
        """Test important points generation when LLM is None."""
        mock_summarizer = Mock()
        mock_summarizer.llm = None

        result = generate_important_points(mock_summarizer, "test summary")

        assert result == ""

    def test_generate_important_points_llm_failure(self):
        """Test important points generation when LLM call fails."""
        mock_summarizer = Mock()
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_summarizer.llm = mock_llm

        result = generate_important_points(mock_summarizer, "test summary")

        assert result == "(Failed to generate important points)"

    def test_generate_application_ideas_success(self):
        """Test successful generation of application ideas."""
        mock_summarizer = Mock()
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "1. Application idea one\n2. Application idea two"
        mock_llm.invoke.return_value = mock_response
        mock_summarizer.llm = mock_llm

        result = generate_application_ideas(mock_summarizer, "test summary")

        assert result == "1. Application idea one\n2. Application idea two"
        mock_llm.invoke.assert_called_once()
        # Verify the call contains a HumanMessage with formatted prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert "test summary" in call_args[0].content

    def test_generate_application_ideas_no_llm(self):
        """Test application ideas generation when LLM is None."""
        mock_summarizer = Mock()
        mock_summarizer.llm = None

        result = generate_application_ideas(mock_summarizer, "test summary")

        assert result == ""

    def test_generate_application_ideas_llm_failure(self):
        """Test application ideas generation when LLM call fails."""
        mock_summarizer = Mock()
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_summarizer.llm = mock_llm

        result = generate_application_ideas(mock_summarizer, "test summary")

        assert result == "(Failed to generate application ideas)"
