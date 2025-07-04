"""
Section header detection and content parsing for docxtract.

This module provides the SectionParser class, which:
- Detects common academic section headers using regex patterns
- Splits raw document text into structured sections for downstream processing

See .github/copilot-instructions.md for documentation and code style standards.
"""

import re
from typing import Dict, List, Optional, Tuple

from .models import DocumentSection, ExtractedDocument


class SectionParser:
    """
    Handles detection and parsing of document sections.

    SECTION_PATTERNS can be customized for different document types.
    """

    # Common section headers in academic papers
    SECTION_PATTERNS: List[Tuple[str, str]] = [
        (r"^Abstract\s*$", "Abstract"),
        (r"^Introduction\s*$", "Introduction"),
        (r"^Methods?\s*$", "Method"),
        (r"^Methodology\s*$", "Methodology"),
        (r"^Results?\s*$", "Results"),
        (r"^Experiments?\s*$", "Experiments"),
        (r"^Discussion\s*$", "Discussion"),
        (r"^Conclusions?\s*$", "Conclusion"),
        (r"^References?\s*$", "References"),
        (r"^Bibliography\s*$", "References"),
    ]

    def __init__(self, section_patterns: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize the section parser.
        Optionally accept custom section patterns for extensibility.
        """
        patterns = (
            section_patterns if section_patterns is not None else self.SECTION_PATTERNS
        )
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), name) for pattern, name in patterns
        ]

    def detect_section_boundaries(self, text: str) -> List[Tuple[str, int]]:
        """
        Detect section boundaries in the text.

        Args:
            text: Full document text

        Returns:
            List of tuples (section_name, line_number)
        """
        lines = text.split("\n")
        boundaries = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            for pattern, section_name in self.compiled_patterns:
                if pattern.match(line):
                    boundaries.append((section_name, i))
                    break

        return boundaries

    def extract_sections(self, text: str) -> List[DocumentSection]:
        """
        Extract structured sections from document text.

        Args:
            text: Full document text

        Returns:
            List of DocumentSection objects
        """
        lines = text.split("\n")
        boundaries = self.detect_section_boundaries(text)

        if not boundaries:
            # No section headers detected; treat the entire document as a single section.
            return [DocumentSection(title="Full Content", content=text)]

        sections = []

        for i, (section_name, start_line) in enumerate(boundaries):
            # Determine end line (start of next section or end of document)
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1][1]
            else:
                end_line = len(lines)

            # Extract content (skip the header line)
            section_lines = lines[start_line + 1 : end_line]

            # Special handling: For Abstract, stop at 'Keywords:'
            if section_name.lower() == "abstract":
                filtered_lines = []
                for line in section_lines:
                    if re.match(r"^keywords\s*:", line.strip(), re.IGNORECASE):
                        break
                    filtered_lines.append(line)
                content = "\n".join(filtered_lines).strip()
            else:
                content = "\n".join(section_lines).strip()

            if content:  # Only add sections with content
                sections.append(DocumentSection(title=section_name, content=content))

        return sections

    def parse_document(self, document: ExtractedDocument) -> ExtractedDocument:
        """
        Parse an extracted document to identify sections.

        Args:
            document: ExtractedDocument with raw content

        Returns:
            ExtractedDocument with parsed sections
        """
        if not document.sections:
            return document

        # Get the full content from the first section (assuming it contains all text)
        full_content = document.sections[0].content
        parsed_sections = self.extract_sections(full_content)

        # Update the document with parsed sections
        document.sections = parsed_sections
        return document
