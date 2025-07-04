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
        Attempt to detect the document title from the text using heuristics.

        Heuristics:
        - Only consider the first 10 lines (titles are usually at the top)
        - Skip empty lines
        - Stop at common section headers (e.g., Abstract, Introduction, Keywords)
        - Stop at lines likely to be author info (e.g., email, affiliations, research group)
        - Only include lines that are reasonably long and not URLs/domains
        - Combine consecutive lines to form a title, but stop if the combined title is long enough

        Args:
            text: Full document text

        Returns:
            Detected title or None if not found
        """
        lines = text.split("\n")
        title_lines = []
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
            # Stop if we hit common section headers
            if line.lower() in ("abstract", "introduction", "keywords"):
                break
            # Stop if line looks like author info (contains @ or common patterns)
            if (
                "@" in line
                or line.lower().startswith(
                    ("author", "email", "university", "department")
                )
                or any(
                    word in line.lower()
                    for word in ["research", "institute", "lab", "group"]
                )
                and len(line) < 50
            ):
                break
            # Add to title if it looks title-like
            if len(line) > 5 and not line.endswith((".com", ".edu", ".org")):
                title_lines.append(line)
                combined = " ".join(title_lines)
                # If we have a reasonable length title, check if next line continues
                if len(combined) > 30 and not line.endswith(":"):
                    break
        if title_lines:
            combined_title = " ".join(title_lines)
            if len(combined_title) < 300:  # Reasonable title length
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
