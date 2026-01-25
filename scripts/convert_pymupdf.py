#!/usr/bin/env python
"""PyMuPDF4LLM PDF to Markdown converter - Fast and lightweight"""

import pymupdf4llm
import pathlib
import argparse


def convert_pdf_to_md(
    pdf_path: str,
    output_path: str | None = None,
    pages: list[int] | None = None,
    write_images: bool = True,
    image_path: str | None = None,
    dpi: int = 150,
    page_chunks: bool = False,
):
    """Convert PDF to Markdown using PyMuPDF4LLM.

    Args:
        pdf_path: Path to the PDF file
        output_path: Output markdown file path (default: same name as PDF with .md extension)
        pages: List of page numbers to process (0-based), None for all pages
        write_images: Whether to extract and save images
        image_path: Directory to save images (default: ./images)
        dpi: Image resolution (default: 150)
        page_chunks: Return list of dicts per page instead of single string

    Returns:
        str or list: Markdown text or list of page chunks
    """
    pdf_file = pathlib.Path(pdf_path)

    if output_path is None:
        output_path = pdf_file.with_suffix(".md")
    else:
        output_path = pathlib.Path(output_path)

    if image_path is None:
        image_path = output_path.parent / "images"
    else:
        image_path = pathlib.Path(image_path)

    # Ensure image directory exists
    if write_images:
        image_path.mkdir(parents=True, exist_ok=True)

    # Convert PDF to Markdown
    md_text = pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=pages,
        write_images=write_images,
        image_path=str(image_path),
        dpi=dpi,
        page_chunks=page_chunks,
    )

    # Save to file (only if not page_chunks mode)
    if not page_chunks:
        output_path.write_bytes(md_text.encode())
        print(f"Conversion complete!")
        print(f"Output: {output_path}")
        print(f"Length: {len(md_text):,} characters")
        if write_images:
            print(f"Images: {image_path}")
    else:
        print(f"Conversion complete! ({len(md_text)} page chunks)")

    return md_text


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using PyMuPDF4LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_pymupdf.py document.pdf
  python convert_pymupdf.py document.pdf -o output.md
  python convert_pymupdf.py document.pdf --pages 0 1 2 --dpi 200
  python convert_pymupdf.py document.pdf --no-images
        """,
    )

    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output markdown file path")
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        help="Page numbers to process (0-based)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't extract images",
    )
    parser.add_argument(
        "--image-path",
        help="Directory to save images",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution (default: 150)",
    )
    parser.add_argument(
        "--page-chunks",
        action="store_true",
        help="Return page chunks with metadata instead of single string",
    )

    args = parser.parse_args()

    convert_pdf_to_md(
        pdf_path=args.pdf_path,
        output_path=args.output,
        pages=args.pages,
        write_images=not args.no_images,
        image_path=args.image_path,
        dpi=args.dpi,
        page_chunks=args.page_chunks,
    )


if __name__ == "__main__":
    main()
