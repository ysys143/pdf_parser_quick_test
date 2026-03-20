#!/usr/bin/env python
"""OpenDataLoader PDF to Markdown converter"""

import opendataloader_pdf
import pathlib
import argparse


def convert_pdf_to_md(pdf_path: str, output_dir: str | None = None):
    """Convert PDF to Markdown using OpenDataLoader PDF.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Output directory (default: ./opendataloader_output)
    """
    pdf_file = pathlib.Path(pdf_path)

    if output_dir is None:
        output_dir = str(pdf_file.parent / "opendataloader_output")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    opendataloader_pdf.convert(
        input_path=[pdf_path],
        output_dir=output_dir,
        format="markdown",
    )

    # Find the generated markdown file
    output_path = pathlib.Path(output_dir)
    md_files = list(output_path.glob("**/*.md"))

    if md_files:
        md_file = md_files[0]
        content = md_file.read_text(encoding="utf-8")
        print("Conversion complete!")
        print(f"Output: {md_file}")
        print(f"Length: {len(content):,} characters")
        return content
    else:
        print("Conversion complete! Output dir: " + output_dir)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using OpenDataLoader PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_opendataloader.py document.pdf
  python convert_opendataloader.py document.pdf -o ./output_dir
        """,
    )

    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output-dir", help="Output directory")

    args = parser.parse_args()

    convert_pdf_to_md(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
