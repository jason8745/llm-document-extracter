# 📄 Copilot Instructions for Document Extractor CLI

## 💡 Project Goal

Create a document extractor system that helps extract knowledge from academic papers. The system will extract structured content from academic PDF documents and provide Chinese summaries for better understanding.

**Phase 1**: Extract content + Chinese summary  
**Phase 2** (Future): RAG storage and retrieval  
**Phase 3** (Future): Personal knowledge base with LLM integration

---

## 🎯 CLI Features (Phase 1)

### 1. Extract PDF Content

- Command format:
  ```bash
  docxtract extract ./paper.pdf --output ./data/paper.md --summary
  ```
- Input: A PDF file (typically academic papers)
- Output: A `.md` file with the following structure:
  - Title (if detected)
  - **Chinese Summary** (using GPT-4.1)
  - Abstract
  - Introduction
  - Method / Methodology
  - Results / Experiments
  - Conclusion / Discussion
  - References (optional)

### 2. Structure Detection

Use simple heuristics or regex to detect section headers:
- Match lines like:
  - `^Abstract$`
  - `^Introduction$`
  - `^Method(s)?$`
  - `^Results$`
  - `^Conclusion$`
  - `^Discussion$`
  - `^References$`

Split and group content based on those headers.

### 3. Markdown Output Format

Save output as readable Markdown file. Example:

```markdown
# Title: A Neural Approach to Paper Extraction

## 中文摘要
這篇論文探討了一種基於神經網絡的論文提取方法。主要貢獻包括：
1. 提出了新的文檔結構識別算法
2. 實現了高精度的內容提取
3. 在多個數據集上驗證了方法的有效性

## Abstract
This paper explores...

## Introduction
In recent years...

## Method
We used...

## Results
Our experiments show...

## Conclusion
We conclude that...

## References
[1] Smith et al. (2020)...
```

---

## 🧰 Suggested Tools / Libraries

Use Python for development.

- PDF parsing: [`PyMuPDF`](https://pymupdf.readthedocs.io/) or [`pdfplumber`](https://github.com/jsvine/pdfplumber)
- CLI framework: [`typer`](https://github.com/tiangolo/typer)
- LLM integration: [`langchain_openai`](https://github.com/langchain-ai/langchain) with `AzureChatOpenAI` for GPT-4.1 Chinese summaries
- Data validation: [`pydantic`](https://pydantic.dev/) v2 for data models and validation
- Markdown writing: built-in file I/O

---

## 🛠️ Suggested File Structure

```text
docxtract/
├── cli.py          # CLI entrypoint
├── extract.py      # PDF parsing and section extraction logic
├── parser.py       # Section header detection
├── summarizer.py   # GPT-4.1 Chinese summary generation
├── writer.py       # Markdown file writer
├── utils.py        # Shared helpers
└── main.py         # Optional top-level runner
```

---

## 🚫 Not Required in This Phase

You do **not** need to:
- Store content in a vector DB
- Perform RAG or retrieval tasks

(These will come in future phases.)

---

## ✅ Deliverables

- A working CLI tool that:
  - Accepts a PDF path
  - Parses and structures its content
  - Generates Chinese summaries using GPT-4.1
  - Outputs a `.md` file following standard paper sections
- Optional bonus:
  - Support for batch processing (e.g., a folder of PDFs)

---

## 🧩 Naming Convention

Tool name: `docxtract`  
Commands:
- `extract`: Parse PDF and output `.md` content

---

## 🧑‍💻 Development Guidelines

### Code Quality Standards
- **Comments**: Add clear but appropriate English comments for all functions, classes, and complex logic
- **Single Responsibility Principle (SRP)**: Each module and function should have one clear responsibility
- **Testability**: Keep code easily testable by avoiding tight coupling and using dependency injection where appropriate

### Required Dependencies
- **LLM Integration**: Use `from langchain_openai import AzureChatOpenAI` for GPT-4.1 integration
- **Data Validation**: Use Pydantic v2 for all data models and validation
- **Type Hints**: Include comprehensive type hints for better code documentation and IDE support

### Project Structure Best Practices
- Separate business logic from I/O operations
- Keep CLI interface thin - delegate to business logic modules
- Use data classes/models for structured data passing between modules
- Implement proper error handling and logging

---

Let Copilot help implement individual files, and suggest ways to improve section detection heuristics or output formatting. Future extensibility should be considered, but the focus is on building a reliable document extractor for academic PDFs.

---

## 📝 Implementation Notes

- Use clear English comments throughout the codebase
- Implement `from langchain_openai import AzureChatOpenAI` for LLM integration
- Apply Pydantic v2 for all data validation and models
- Follow Single Responsibility Principle (SRP) to keep modules focused
- Design code to be easily testable with minimal dependencies
