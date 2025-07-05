"""
LangChain LCEL pipeline for document extraction and summarization.

This pipeline orchestrates the following steps:
1. Extract PDF
2. Parse sections
3. LLM summarize sections
4. LLM summarize overall
5. LLM read overall summaries and generate top-5 important points and some ideas
6. Write to markdown

Markdown output includes:
- Title
- Top-5 important points
- Application ideas
- Chinese summary

See .github/copilot-instructions.md for prompt and output standards.
"""

from pathlib import Path
from typing import Any, Dict

from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableLambda

from .extract import PDFExtractor
from .parser import SectionParser
from .summarizer import ChineseSummarizer


def extract_pdf_step(pdf_path: Path) -> Dict[str, Any]:
    """Extract text and metadata from the PDF file."""
    extractor = PDFExtractor()
    try:
        doc = extractor.extract_document(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF: {e}")
    return {"document": doc}


def parse_sections_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the document into logical sections."""
    parser = SectionParser()
    try:
        doc = parser.parse_document(inputs["document"])
    except Exception as e:
        raise RuntimeError(f"Failed to parse sections: {e}")
    return {"document": doc}


def summarize_sections_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize each section using the LLM."""
    summarizer = ChineseSummarizer()
    doc = inputs["document"]
    try:
        section_summaries = summarizer.summarize_all_sections(doc)
    except Exception as e:
        raise RuntimeError(f"Failed to summarize sections: {e}")
    return {"document": doc, "section_summaries": section_summaries}


def summarize_overall_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an overall summary from all section summaries."""
    summarizer = ChineseSummarizer()
    doc = inputs["document"]
    section_summaries = inputs["section_summaries"]
    try:
        overall_summary = summarizer.summarize_overall(section_summaries, doc.title)
    except Exception as e:
        overall_summary = "(Failed to generate overall summary)"
    return {
        "document": doc,
        "section_summaries": section_summaries,
        "overall_summary": overall_summary,
    }


def load_prompt_template(path: Path) -> str:
    """Load prompt template from file."""
    return path.read_text(encoding="utf-8").strip()


def get_take_away_prompt(summary: str) -> str:
    template = load_prompt_template(
        Path(__file__).parent / "prompts" / "take_away_prompt.md"
    )
    return template.format(summary=summary)


def get_ideas_prompt(summary: str) -> str:
    template = load_prompt_template(
        Path(__file__).parent / "prompts" / "generate_ideas_prompt.md"
    )
    return template.format(summary=summary)


def generate_important_points(
    summarizer: ChineseSummarizer, overall_summary: str
) -> str:
    """Generate important points using the LLM."""
    try:
        if not summarizer.llm:
            return ""

        points_prompt = get_take_away_prompt(overall_summary)
        response = summarizer.llm.invoke([HumanMessage(content=points_prompt)])
        return response.content.strip()
    except Exception:
        return "(Failed to generate important points)"


def generate_application_ideas(
    summarizer: ChineseSummarizer, overall_summary: str
) -> str:
    """Generate application ideas using the LLM."""
    try:
        if not summarizer.llm:
            return ""

        ideas_prompt = get_ideas_prompt(overall_summary)
        response = summarizer.llm.invoke([HumanMessage(content=ideas_prompt)])
        return response.content.strip()
    except Exception:
        return "(Failed to generate application ideas)"


def important_points_and_ideas_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate top-5 important points and application ideas using the LLM."""
    summarizer = ChineseSummarizer()
    doc = inputs["document"]
    overall_summary = inputs["overall_summary"]

    points = generate_important_points(summarizer, overall_summary)
    ideas = generate_application_ideas(summarizer, overall_summary)

    return {
        "document": doc,
        "section_summaries": inputs["section_summaries"],
        "overall_summary": overall_summary,
        "important_points": points,
        "ideas": ideas,
    }


def to_markdown_step(inputs: Dict[str, Any]) -> str:
    """Assemble the final Markdown output from all pipeline results."""
    doc = inputs["document"]
    lines = []
    if doc.title:
        lines.append(f"# {doc.title}")
        lines.append("")
    if inputs.get("important_points"):
        lines.append("## Top-5 Important Points")
        lines.append("")
        lines.append(inputs["important_points"])
        lines.append("")
    if inputs.get("ideas"):
        lines.append("## Application Ideas")
        lines.append("")
        lines.append(inputs["ideas"])
        lines.append("")
    lines.append("## Chinese Summary")
    lines.append("")
    lines.append(
        inputs.get("overall_summary") or "(Failed to generate overall summary)"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Chinese summary generated using GPT-4.1*")
    return "\n".join(lines)


# LCEL pipeline definition
DocExtractPipeline = (
    RunnableLambda(extract_pdf_step)
    | RunnableLambda(parse_sections_step)
    | RunnableLambda(summarize_sections_step)
    | RunnableLambda(summarize_overall_step)
    | RunnableLambda(important_points_and_ideas_step)
    | RunnableLambda(to_markdown_step)
)
