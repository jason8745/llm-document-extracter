"""CLI interface for docxtract using Typer.

This CLI provides the main entry point for extracting and summarizing academic PDFs as Markdown summaries.
Output includes: title, top-5 points, application ideas, and a Chinese summary.
See .github/copilot-instructions.md for prompt and output standards.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .chain import DocExtractPipeline

# Initialize Typer app and Rich console
app = typer.Typer(
    help="Document extractor for academic papers. Output is Markdown summary (Chinese content, English section headers). See .github/copilot-instructions.md for standards."
)
console = Console()


@app.command(
    name="summary",
    help="Extract and summarize an academic PDF as Markdown. Output includes title, top-5 points, application ideas, and a Chinese summary. See .github/copilot-instructions.md for standards.",
)
def summary(
    pdf_path: Path = typer.Argument(
        ...,
        help="Path to the PDF file to extract",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for the Markdown file"
    ),
    preview: bool = typer.Option(
        False, "--preview", "-p", help="Preview the output without saving to file"
    ),
) -> None:
    """
    Extract and summarize an academic PDF, outputting a Markdown summary.
    The summary and section content will be in Chinese; CLI and section headers are in English.
    Output includes: title, top-5 points, application ideas, and a Chinese summary.
    See .github/copilot-instructions.md for prompt and output standards.
    """
    # Determine output path if not provided
    if not output and not preview:
        output = pdf_path.with_suffix(".md")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Run the summary pipeline
            task = progress.add_task("Generating Chinese summary...", total=None)
            try:
                summary_md = DocExtractPipeline.invoke(pdf_path)
                if summary_md:
                    if not preview:
                        with open(output, "w", encoding="utf-8") as f:
                            f.write(summary_md)
                        progress.update(task, description=f"✓ Saved to {output}")
                        console.print(
                            f"\n[green]✓ Successfully extracted to: {output}[/green]"
                        )
                    else:
                        console.print("\n[bold]Preview of summary markdown:[/bold]")
                        console.print(summary_md)
                    progress.update(
                        task, description="✓ Chinese summary generated (pipeline)"
                    )
                    return
                else:
                    progress.update(task, description="⚠ Summary generation failed")
                    console.print(
                        "[red]Summary generation failed. Please check your input file and environment variables.[/red]"
                    )
            except Exception as e:
                progress.update(task, description="⚠ Summary generation failed")
                console.print(
                    f"[red]Error in pipeline: {str(e)}[/red]\n[dim]Check your input file and environment variables.[/dim]"
                )
            # Only summary mode is supported; exit after summary
    except Exception as e:
        console.print(
            f"[red]Error: {str(e)}[/red]\n[dim]If this persists, please report an issue and include the error message above.[/dim]"
        )
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"docxtract version {__version__}")


if __name__ == "__main__":
    app()
