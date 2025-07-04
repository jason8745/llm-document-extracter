"""
PDF parsing and text extraction functionality for docxtract.

This module provides the PDFExtractor class, which handles:
- Reading and extracting text from PDF files using PyMuPDF
- Heuristic detection of the document title from extracted text
- Wrapping extracted content into structured data models for downstream processing

See .github/copilot-instructions.md for documentation and code style standards.
"""

from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from .models import DocumentSection, ExtractedDocument


class PDFExtractor:
    """Handles PDF parsing and text extraction."""

    def __init__(self):
        """Initialize the PDF extractor."""
        pass

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract all text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Full text content of the PDF

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF cannot be read
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += text + "\n"

            doc.close()
            return full_text.strip()

        except Exception as e:
            raise Exception(
                f"Failed to extract text from PDF: {str(e)}. Check if the file is corrupted, encrypted, or not a valid PDF."
            )

    def detect_title(self, text: str) -> Optional[str]:
        """
        Heuristic-based title detection from the top of the document.
        - Only consider the first 3 non-empty lines
        - Stop at section headers, author/affiliation/keywords lines, or single name lines
        - Only return title if length is between 10 and 120 chars (inclusive)
        """
        import re

        lines = text.split("\n")
        title_lines = []
        non_empty_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            non_empty_count += 1
            lower = line.lower()

            # Stop at section headers
            if lower in ("abstract", "introduction", "keywords"):
                break

            if re.match(r"^keywords\s*:?", line, re.IGNORECASE):
                break

            # If line is likely author/affiliation, break but do not append
            if (
                "@" in line
                or lower.startswith(("author", "email", "university", "department"))
                or any(word in lower for word in ["research", "institute", "lab", "group"])
                or re.match(r"^authors?\s*:?", line, re.IGNORECASE)
                or re.match(r"^[a-zA-Z .,'-]+\d?(,\s*[a-zA-Z .,'-]+\d?)+$", line)  # author list
                or ("," in line and len(line.split(",")) > 1 and len(line) < 80)
                or re.match(r"^[A-Z][a-zA-Z'`-]+ [A-Z][a-zA-Z'`-]+$", line)  # single name
            ):
                break

            # Accept title-like lines
            if len(line) > 5 and not line.endswith((".com", ".edu", ".org")):
                title_lines.append(line)

            if non_empty_count >= 3:
                break

        if title_lines:
            combined_title = " ".join(title_lines).strip()
            if 10 <= len(combined_title) <= 120:
                return combined_title

        return None

    def extract_document(self, pdf_path: Path) -> ExtractedDocument:
        """
        Extract complete document structure from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractedDocument with basic text extraction and detected title
        """
        full_text = self.extract_text_from_pdf(pdf_path)
        title = self.detect_title(full_text)

        # Create a single section with all content (section parsing is handled downstream)
        main_section = DocumentSection(title="Full Content", content=full_text)

        return ExtractedDocument(
            title=title, sections=[main_section], source_file=str(pdf_path)
        )
