"""Unit tests for docxtract.summarizer module."""

from unittest.mock import Mock, patch

import pytest
from langchain.schema import AIMessage

from docxtract.models import ExtractedDocument, DocumentSection, SummaryRequest
from docxtract.summarizer import ChineseSummarizer


class TestChineseSummarizer:
    """Test cases for ChineseSummarizer class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = Mock()
        mock.invoke.return_value = AIMessage(content="Mock summary in Chinese")
        return mock

    @pytest.fixture
    def summarizer(self, mock_llm):
        """Create a ChineseSummarizer with mocked LLM."""
        with patch("docxtract.summarizer.AzureChatOpenAI") as mock_azure:
            mock_azure.return_value = mock_llm
            summarizer = ChineseSummarizer()
            summarizer.llm = mock_llm
            return summarizer

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        sections = [
            DocumentSection(title="Abstract", content="This paper presents a novel approach to machine learning."),
            DocumentSection(title="Introduction", content="Machine learning has revolutionized many fields."),
            DocumentSection(title="Method", content="We propose a new neural network architecture."),
            DocumentSection(title="Results", content="Our experiments show significant improvements."),
            DocumentSection(title="Conclusion", content="This work opens new research directions."),
        ]
        return ExtractedDocument(
            title="Novel ML Approach",
            sections=sections,
            source_file="paper.pdf"
        )

    @pytest.fixture
    def sample_summary_request(self):
        """Create a sample summary request."""
        return SummaryRequest(
            document_content="This is a test document content for summarization.",
            title="Test Paper Title",
            focus_areas=["methodology", "results", "contributions"]
        )

    def test_init_with_env_vars(self):
        """Test ChineseSummarizer initialization with environment variables."""
        with patch.dict("os.environ", {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4"
        }):
            with patch("docxtract.summarizer.AzureChatOpenAI") as mock_azure:
                summarizer = ChineseSummarizer()
                
                mock_azure.assert_called_once_with(
                    azure_endpoint="https://test.openai.azure.com/",
                    api_key="test_key",
                    api_version="2024-02-01",
                    azure_deployment="gpt-4",
                    temperature=0.3,
                    max_tokens=4096
                )

    def test_init_missing_env_vars(self):
        """Test ChineseSummarizer initialization with missing environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            summarizer = ChineseSummarizer()
            
            # Should create summarizer with llm=None when env vars are missing
            assert summarizer.llm is None

    def test_select_core_sections(self):
        """Test core section selection."""
        sections = [
            DocumentSection(title="Abstract", content="Abstract content"),
            DocumentSection(title="Introduction", content="Intro content"),
            DocumentSection(title="Related Work", content="Related work content"),
            DocumentSection(title="Method", content="Method content"),
            DocumentSection(title="Implementation", content="Implementation content"),
            DocumentSection(title="Results", content="Results content"),
            DocumentSection(title="Conclusion", content="Conclusion content"),
            DocumentSection(title="References", content="References content"),
        ]
        
        core_sections = ChineseSummarizer.select_core_sections(sections)
        core_titles = [sec.title for sec in core_sections]
        
        # Should include priority sections
        assert "Abstract" in core_titles
        assert "Introduction" in core_titles
        assert "Method" in core_titles
        assert "Results" in core_titles
        assert "Conclusion" in core_titles
        
        # Should exclude non-priority sections
        assert "Related Work" not in core_titles
        assert "Implementation" not in core_titles
        assert "References" not in core_titles

    def test_section_prompts_exist(self):
        """Test that section prompts are defined for priority sections."""
        for section in ChineseSummarizer.PRIORITY_SECTIONS:
            assert section in ChineseSummarizer.SECTION_PROMPTS
            assert isinstance(ChineseSummarizer.SECTION_PROMPTS[section], str)
            assert len(ChineseSummarizer.SECTION_PROMPTS[section]) > 0

    def test_summarize_section_with_specific_prompt(self, summarizer, mock_llm):
        """Test section summarization with section-specific prompt."""
        result = summarizer.summarize_section("Abstract", "This is an abstract about AI research.", "Test Paper")
        
        # Should call LLM with section-specific prompt
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        # Check that the prompt contains section-specific instructions
        prompt_content = call_args[0].content
        assert "研究動機" in prompt_content  # From Abstract prompt
        assert "This is an abstract about AI research." in prompt_content
        assert "Test Paper" in prompt_content
        
        assert result == "Mock summary in Chinese"

    def test_summarize_section_with_generic_prompt(self, summarizer, mock_llm):
        """Test section summarization with generic prompt for unknown sections."""
        result = summarizer.summarize_section("Unknown Section", "This is unknown content.", "Test Paper")
        
        # Should call LLM with generic prompt
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        prompt_content = call_args[0].content
        assert "Unknown Section" in prompt_content
        assert "This is unknown content." in prompt_content
        
        assert result == "Mock summary in Chinese"

    def test_summarize_section_empty_content(self, summarizer):
        """Test section summarization with empty content."""
        result = summarizer.summarize_section("Abstract", "", "Test Paper")
        
        # Should return empty string for empty content
        assert result == ""

    def test_summarize_section_no_llm(self):
        """Test section summarization when LLM is None."""
        summarizer = ChineseSummarizer.__new__(ChineseSummarizer)
        summarizer.llm = None
        
        result = summarizer.summarize_section("Abstract", "Some content", "Test Paper")
        
        # Should return empty string when no LLM
        assert result == ""

    def test_summarize_all_sections(self, summarizer, mock_llm, sample_document):
        """Test summarizing all sections in a document."""
        result = summarizer.summarize_all_sections(sample_document)
        
        # Should call LLM for each core section
        assert mock_llm.invoke.call_count == 5  # 5 core sections in sample
        
        # Should return summaries for each section
        assert len(result) == 5
        assert "Abstract" in result
        assert "Introduction" in result
        assert "Method" in result
        assert "Results" in result
        assert "Conclusion" in result
        
        for section_summary in result.values():
            assert section_summary == "Mock summary in Chinese"

    def test_summarize_overall(self, summarizer, mock_llm):
        """Test overall summary generation."""
        section_summaries = {
            "Abstract": "摘要部分的中文總結",
            "Method": "方法部分的中文總結",
            "Results": "結果部分的中文總結"
        }
        
        result = summarizer.summarize_overall(section_summaries, "Test Paper")
        
        # Should call LLM with overall summary prompt
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        prompt_content = call_args[0].content
        assert "摘要部分的中文總結" in prompt_content
        assert "方法部分的中文總結" in prompt_content
        assert "結果部分的中文總結" in prompt_content
        
        assert result == "Mock summary in Chinese"

    def test_generate_summary(self, summarizer, mock_llm, sample_document):
        """Test generating summary from document."""
        # Mock different responses for different calls
        mock_responses = ["Section summary"] * 5 + ["Overall summary"]
        mock_llm.invoke.side_effect = [AIMessage(content=resp) for resp in mock_responses]
        
        result = summarizer.generate_summary(sample_document)
        
        # Should call LLM for sections + overall
        assert mock_llm.invoke.call_count == 6  # 5 sections + 1 overall
        
        # Should return markdown document with paper title
        assert "# Novel ML Approach" in result
        assert "## 中文摘要" in result
        assert "Overall summary" in result
        assert "*Chinese summary generated using GPT-4.1, core-section summarization mode*" in result

    def test_is_configured(self, summarizer):
        """Test configuration check."""
        # Should be configured with mock LLM
        assert summarizer.is_configured() is True
        
        # Should not be configured without LLM
        summarizer.llm = None
        assert summarizer.is_configured() is False

    def test_generate_top_takeaways(self, summarizer, mock_llm):
        """Test generating top takeaways."""
        summary = "這是一篇關於機器學習的論文"
        
        result = summarizer.generate_top_takeaways(summary)
        
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        prompt_content = call_args[0].content
        assert "最重要的重點" in prompt_content
        assert "條列方式" in prompt_content
        assert summary in prompt_content
        
        assert result == "Mock summary in Chinese"

    def test_generate_extended_applications(self, summarizer, mock_llm):
        """Test generating extended applications."""
        summary = "這是一篇關於機器學習的論文"
        
        result = summarizer.generate_extended_applications(summary)
        
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        prompt_content = call_args[0].content
        assert "應用方向" in prompt_content
        assert "後續研究發展" in prompt_content
        assert summary in prompt_content
        
        assert result == "Mock summary in Chinese"

    def test_llm_error_handling(self, mock_llm):
        """Test LLM error handling during initialization."""
        with patch("docxtract.summarizer.AzureChatOpenAI", side_effect=Exception("LLM initialization failed")):
            with patch.dict("os.environ", {
                "AZURE_OPENAI_API_KEY": "test_key",
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                "AZURE_OPENAI_API_VERSION": "2024-02-01",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4"
            }):
                with pytest.raises(Exception) as exc_info:
                    ChineseSummarizer()
                
                assert "LLM initialization failed" in str(exc_info.value)

    def test_summarize_section_llm_error(self, summarizer, mock_llm):
        """Test section summarization with LLM error."""
        mock_llm.invoke.side_effect = Exception("LLM call failed")
        
        result = summarizer.summarize_section("Abstract", "Test content", "Test Paper")
        
        # Should return empty string on error
        assert result == ""

    @pytest.mark.parametrize("section_title", ChineseSummarizer.PRIORITY_SECTIONS)
    def test_all_priority_sections_have_prompts(self, section_title):
        """Test that all priority sections have corresponding prompts."""
        assert section_title in ChineseSummarizer.SECTION_PROMPTS
        prompt = ChineseSummarizer.SECTION_PROMPTS[section_title]
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0
        assert "繁體中文" in prompt or "中文" in prompt  # Should be Chinese prompt
