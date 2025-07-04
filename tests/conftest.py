"""
Pytest configuration and fixtures for docxtract tests.

This module provides common fixtures and test utilities used across all test modules.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from docxtract.models import DocumentSection, ExtractedDocument, SummaryRequest


@pytest.fixture
def sample_document_section():
    """Create a sample DocumentSection for testing."""
    return DocumentSection(
        title="Abstract",
        content="This is a sample abstract content for testing purposes.",
        page_numbers=[1]
    )


@pytest.fixture
def sample_extracted_document():
    """Create a sample ExtractedDocument for testing."""
    sections = [
        DocumentSection(
            title="Abstract",
            content="This paper presents a novel approach to document extraction.",
            page_numbers=[1]
        ),
        DocumentSection(
            title="Introduction",
            content="Document processing has become increasingly important.",
            page_numbers=[1, 2]
        ),
        DocumentSection(
            title="Method",
            content="We propose a hybrid approach combining PDF parsing and NLP.",
            page_numbers=[2, 3]
        ),
        DocumentSection(
            title="Results",
            content="Our method achieves 95% accuracy on test documents.",
            page_numbers=[3, 4]
        ),
        DocumentSection(
            title="Conclusion",
            content="The proposed method shows significant improvements.",
            page_numbers=[4]
        ),
    ]
    
    return ExtractedDocument(
        title="A Novel Approach to Document Extraction",
        summary_zh="這篇論文提出了一種新的文檔提取方法，結合了PDF解析和自然語言處理技術。",
        sections=sections,
        source_file="/path/to/test/document.pdf"
    )


@pytest.fixture
def sample_summary_request():
    """Create a sample SummaryRequest for testing."""
    return SummaryRequest(
        document_content="This is sample document content for testing summary generation.",
        title="Test Document",
        focus_areas=["methodology", "results", "contributions"]
    )


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_pdf_content():
    """Sample PDF text content for testing."""
    return """
    A Novel Approach to Document Extraction
    
    Abstract
    This paper presents a novel approach to document extraction using advanced PDF parsing
    techniques combined with natural language processing. Our method achieves high accuracy
    in extracting structured content from academic papers.
    
    Introduction
    Document processing has become increasingly important in the digital age. Many documents
    are stored as PDFs, making extraction challenging. This work addresses these challenges.
    
    Method
    We propose a hybrid approach that combines:
    1. Advanced PDF parsing using PyMuPDF
    2. Section detection using regex patterns
    3. Content structuring using Pydantic models
    
    Results
    Our method achieves 95% accuracy on a test set of 100 academic papers. The extraction
    time is significantly reduced compared to manual processing.
    
    Conclusion
    The proposed method shows significant improvements over existing approaches. Future work
    will focus on extending support to additional document types.
    
    References
    [1] Smith et al. (2020). PDF Processing Techniques.
    [2] Jones et al. (2021). Natural Language Processing for Documents.
    """


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing Chinese summarization."""
    return {
        "abstract": "這是一個關於文檔提取的研究，提出了新的方法。",
        "introduction": "介紹了數字時代文檔處理的重要性。",
        "method": "提出了結合PDF解析和自然語言處理的混合方法。",
        "results": "在100篇學術論文的測試集上達到95%的準確率。",
        "conclusion": "該方法相比現有方法有顯著改進。",
        "overall": "這篇論文提出了一種新的文檔提取方法，結合了PDF解析和自然語言處理技術，在準確率和效率方面都有顯著提升。"
    }


@pytest.fixture
def mock_azure_openai_config():
    """Mock Azure OpenAI configuration for testing."""
    return {
        "azure_endpoint": "https://test-openai.openai.azure.com/",
        "api_key": "test-api-key-12345",
        "api_version": "2024-02-15-preview",
        "deployment_name": "gpt-4-turbo"
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Set test environment variables
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-openai.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4-turbo")


@pytest.fixture
def sample_section_headers():
    """Sample section headers for testing parser."""
    return [
        "Abstract",
        "1. Introduction",
        "2. Related Work",
        "3. Methodology",
        "4. Experiments",
        "5. Results",
        "6. Discussion",
        "7. Conclusion",
        "References",
    ]


# Pytest markers for different test categories
pytestmark = pytest.mark.unit
