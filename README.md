# # Document Extractor (docxtract)

A CLI tool for extracting structured content from academic PDF documents and generating high-quality Chinese summaries.

## Features

- 📄 Extract text content from PDF files
- 🔍 Automatically detect and parse academic paper sections
- 🇹🇼 Generate Chinese summaries using GPT-4.1 (per-section summarization, with specialized prompts)
- 📝 Output structured content as Markdown files (including per-section and overall summary)
- 🚀 Simple CLI interface with rich progress indicators

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd llm-document-extracter
```

2. Install dependencies using uv:

```bash
uv sync
```

## Usage

### Generate Chinese Summary (Markdown Output)

To generate high-quality Chinese summaries, you need to configure Azure OpenAI:

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Edit `.env` and add your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
```

3. Generate a summary (Markdown output):

```bash
python main.py summary research_paper.pdf -o summaries/paper.md
```

- Each section (Abstract, Introduction, Method, Results, Conclusion, etc.) is summarized separately to avoid token overflow and improve quality.
- Specialized prompts are used for key sections (e.g., Abstract, Conclusion) to extract more valuable information.
- All section summaries are then combined to generate an overall Chinese summary.
- The output is a Markdown file with the following structure:

```markdown
# Paper Title

## 中文摘要
(Overall summary generated from all section summaries)

## Abstract
(Abstract section summary)

## Introduction
(Introduction section summary)

## Method
(Method section summary)

## Results
(Results section summary)

## Conclusion
(Conclusion section summary)

---
*Extracted from: path/to/source.pdf*
*Chinese summary generated using GPT-4.1, per-section summarization mode*
```

### Command Options

```bash
python main.py summary [OPTIONS] PDF_PATH

Options:
  -o, --output PATH    Output path for the Markdown file
  -p, --preview        Preview the output without saving to file
  --help              Show help message
```

### Examples

```bash
# Generate summary and save to default output path
python main.py summary research_paper.pdf

# Generate summary with custom output path
python main.py summary research_paper.pdf -o summaries/paper.md

# Preview summary markdown without saving
python main.py summary research_paper.pdf --preview
```

## Output Format

The tool generates Markdown files with the following structure:

```markdown
# Paper Title

## 中文摘要
(Overall summary generated from all section summaries)

## Abstract
(Abstract section summary)

## Introduction
(Introduction section summary)

## Method
(Method section summary)

## Results
(Results section summary)

## Conclusion
(Conclusion section summary)

---
*Extracted from: path/to/source.pdf*
*Chinese summary generated using GPT-4.1, per-section summarization mode*
```

## Project Structure

```
docxtract/
├── cli.py          # CLI interface using Typer
├── extract.py      # PDF text extraction with PyMuPDF
├── parser.py       # Section header detection and parsing
├── summarizer.py   # Chinese summary generation with Azure OpenAI (per-section logic)
├── writer.py       # Markdown file output
├── models.py       # Pydantic data models
└── utils.py        # Shared utilities
```

## Useful CLI Commands

- `make sync` &nbsp;&nbsp;&nbsp;&nbsp;# Initialize project environment (install dependencies, setup venv)
- `make dev` &nbsp;&nbsp;&nbsp;&nbsp;# Run server locally (FastAPI, hot reload)
- `make lint` &nbsp;&nbsp;&nbsp;&nbsp;# Check code lint using ruff
- `make format` &nbsp;&nbsp;&nbsp;&nbsp;# Format code using ruff
- `make check` &nbsp;&nbsp;&nbsp;&nbsp;# Check code typing using pyright
- `make clean` &nbsp;&nbsp;&nbsp;&nbsp;# Clean Python cache files
- `make unit-test` &nbsp;&nbsp;&nbsp;&nbsp;# Run unit tests
- `make integration-test` &nbsp;&nbsp;&nbsp;&nbsp;# Run integration tests

## Development

### Code Quality

The project follows these standards:

- Type hints for all functions
- Pydantic v2 for data validation
- Single Responsibility Principle (SRP)
- Clear English comments
- Comprehensive error handling

## Requirements

- Python 3.12+
- PyMuPDF for PDF parsing
- Typer for CLI interface
- Pydantic v2 for data validation
- LangChain OpenAI for summary generation
- Rich for beautiful CLI output

## Future Development

- **Phase 2**: RAG storage and retrieval
- **Phase 3**: Personal knowledge base with LLM integration
- Batch processing for multiple PDFs
- Enhanced section detection algorithms
- Support for different document types

## License

See LICENSE file for details.
