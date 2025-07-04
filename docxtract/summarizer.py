"""
Chinese summary generation for docxtract using GPT-4.1 via Azure OpenAI.

This module provides the ChineseSummarizer class, which:
- Generates per-section and overall Chinese summaries for academic papers
- Uses specialized prompts for each section and summary type
- Supports top takeaways and extended application generation

See .github/copilot-instructions.md for prompt and documentation standards.
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()  # Auto-load .env at import time
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI

from .models import ExtractedDocument, SummaryRequest


class ChineseSummarizer:
    """Handles Chinese summary generation using GPT-4.1, with per-section summarization and specialized prompts."""

    PRIORITY_SECTIONS = ["Abstract", "Conclusion", "Introduction", "Method", "Results"]

    @classmethod
    def select_core_sections(cls, sections):
        """
        Select only core sections for summarization.
        """
        return [sec for sec in sections if sec.title in cls.PRIORITY_SECTIONS]

    # Section-specific prompts for LLM (Traditional Chinese)
    SECTION_PROMPTS = {
        "Abstract": (
            "請將以下 Abstract 內容以繁體中文摘要，強調：\n"
            "- 研究動機\n"
            "- 主要研究問題\n"
            "- 方法概述\n"
            "- 主要發現與貢獻\n"
            "摘要應簡明扼要，適合學術讀者快速掌握論文核心。"
        ),
        "Introduction": (
            "請將以下 Introduction 內容以繁體中文摘要，強調：\n"
            "- 研究背景與動機\n"
            "- 相關領域現況\n"
            "- 研究目的與重要性\n"
            "- 主要貢獻\n"
            "摘要應清楚說明本研究的定位與價值。"
        ),
        "Method": (
            "請將以下 Method 內容以繁體中文摘要，強調：\n"
            "- 研究設計與架構\n"
            "- 主要方法或技術\n"
            "- 實驗流程或數據來源\n"
            "- 與現有方法的差異\n"
            "摘要應讓讀者能快速理解本研究的技術核心。"
        ),
        "Results": (
            "請將以下 Results 內容以繁體中文摘要，強調：\n"
            "- 主要實驗結果\n"
            "- 數據或指標\n"
            "- 與預期或基線的比較\n"
            "- 重要發現\n"
            "摘要應聚焦於最具代表性的成果。"
        ),
        "Conclusion": (
            "請將以下 Conclusion 內容以繁體中文摘要，強調：\n"
            "- 研究結論\n"
            "- 主要貢獻與意義\n"
            "- 研究限制\n"
            "- 未來展望\n"
            "摘要應協助讀者掌握本研究的總結與後續方向。"
        ),
    }
    DEFAULT_PROMPT = (
        "請將以下章節內容以繁體中文摘要，重點整理章節核心內容，適合學術背景讀者。"
    )

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
    ):
        """
        Initialize the Chinese summarizer.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version to use
            deployment_name: Azure deployment name for GPT-4.1
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Initialize the Azure ChatOpenAI client
        if (
            self.azure_endpoint
            and self.api_key
            and self.api_version
            and self.deployment_name
        ):
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                azure_deployment=self.deployment_name,
                temperature=0.3,
                max_tokens=4096,
            )
        else:
            self.llm = None

    def _create_summary_prompt(self, request: SummaryRequest) -> str:
        """
        Create a prompt for Chinese summary generation.

        Args:
            request: Summary request with document content

        Returns:
            Formatted prompt for the LLM
        """
        title_part = f"論文標題：{request.title}\n\n" if request.title else ""
        focus_areas = "、".join(request.focus_areas)

        prompt = f"""請為以下學術論文提供一個簡潔的中文摘要。

{title_part}請重點關注：{focus_areas}

論文內容：
{request.document_content[:4000]}  # Limit content length for token management

請以以下格式提供摘要：
這篇論文探討了...主要貢獻包括：
1. [第一個貢獻]
2. [第二個貢獻] 
3. [第三個貢獻]

摘要應該：
- 用繁體中文撰寫
- 突出主要方法和發現
- 保持簡潔但信息豐富
- 適合學術背景的讀者
"""
        return prompt

    def summarize_section(
        self, section_title: str, section_content: str, paper_title: str = None
    ) -> str:
        """
        Summarize a single section using a specialized or default prompt.
        """
        if not self.llm or not section_content.strip():
            return ""
        prompt = self.SECTION_PROMPTS.get(section_title, self.DEFAULT_PROMPT)
        title_part = f"論文標題：{paper_title}\n\n" if paper_title else ""
        full_prompt = f"{prompt}\n\n{title_part}章節：{section_title}\n內容：\n{section_content[:3000]}"
        message = HumanMessage(content=full_prompt)
        try:
            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error summarizing section {section_title}: {str(e)}")
            return ""

    def summarize_all_sections(self, document: ExtractedDocument) -> Dict[str, str]:
        """
        Summarize only core sections and return a dict: {section_title: summary}
        """
        summaries: Dict[str, str] = {}
        core_sections = self.select_core_sections(document.sections)
        for section in core_sections:
            summary = self.summarize_section(
                section.title, section.content, document.title
            )
            summaries[section.title] = summary
        return summaries

    def summarize_overall(
        self, section_summaries: Dict[str, str], paper_title: str = None
    ) -> str:
        """
        Generate an overall summary from all section summaries.
        """
        if not self.llm:
            return ""
        # Combine all section summaries
        combined = "\n\n".join(
            [f"{k}:\n{v}" for k, v in section_summaries.items() if v]
        )
        prompt = "請根據下列各章節的繁體中文摘要，彙整出一份完整的論文總結，強調研究動機、方法、主要發現與貢獻，適合學術背景讀者。"
        title_part = f"論文標題：{paper_title}\n\n" if paper_title else ""
        full_prompt = f"{prompt}\n\n{title_part}章節摘要彙整如下：\n{combined[:6000]}"
        message = HumanMessage(content=full_prompt)
        try:
            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error generating overall summary: {str(e)}")
            return ""

    def generate_summary(self, document: ExtractedDocument) -> str:
        """
        Summarize only core sections, then generate an overall summary, and return Markdown string.
        The output Markdown's title will always be the paper's title.
        """
        if not self.llm:
            print("Warning: Azure OpenAI not configured. Skipping summary generation.")
            return ""
        # 1. Summarize only core sections
        section_summaries = self.summarize_all_sections(document)
        # 2. Generate overall summary
        overall_summary = self.summarize_overall(section_summaries, document.title)
        # 3. Assemble Markdown output
        lines = []
        paper_title = document.title or "(Unknown Title)"
        lines.append(f"# {paper_title}")
        lines.append("")
        lines.append("## 中文摘要")
        lines.append("")
        lines.append(overall_summary or "(總體摘要產生失敗)")
        lines.append("")
        for section in document.sections:
            if section.title in section_summaries:
                lines.append(f"## {section.title}")
                lines.append("")
                lines.append(
                    section_summaries.get(section.title, "(本章節摘要產生失敗)")
                )
                lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*Extracted from: {document.source_file}*")
        lines.append(
            "*Chinese summary generated using GPT-4.1, core-section summarization mode*"
        )
        return "\n".join(lines)

    def is_configured(self) -> bool:
        """Check if the summarizer is properly configured."""
        return self.llm is not None

    def generate_top_takeaways(self, summary: str) -> str:
        """
        Generate top-5 important takeaways from the summary using the LLM.
        """
        prompt = f"""以下是論文的總體摘要內容，請從中整理出五個最重要的重點：

        摘要內容：
        {summary[:3000]}

        請以條列方式回答：
        1. 
        2. 
        3. 
        4. 
        5. 
        """
        try:
            return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            print(f"Error generating top takeaways: {e}")
            return "(重點摘要產生失敗)"

    def generate_extended_applications(self, summary: str) -> str:
        """
        Generate possible application directions or future research ideas from the summary using the LLM.
        """
        prompt = f"""你是一位具備研究與創新能力的AI助手，請根據下列論文摘要，提出可能的應用方向或後續研究發展。

        摘要內容：
        {summary[:3000]}

        請以條列方式提供：
        1. 
        2. 
        3. 
        """
        try:
            return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            print(f"Error generating extended applications: {e}")
            return "(應用發想產生失敗)"
