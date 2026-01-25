#!/usr/bin/env python
"""MinerU PDF to Markdown converter - MD only output"""

import os
# Force CPU to avoid MPS limitations on Apple Silicon - MUST be set before torch import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA

# Disable MPS completely by patching torch before other imports
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from mineru.cli.common import do_parse
from pathlib import Path


def convert_pdf_to_md(pdf_path: str, output_dir: str = "./mineru_md_only", lang: str = "en"):
    """Convert PDF to Markdown using MinerU with minimal output files.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Output directory for the markdown file
        lang: Language code (en, ch, korean, japan, etc.)
    """
    pdf_bytes = Path(pdf_path).read_bytes()

    do_parse(
        output_dir=output_dir,
        pdf_file_names=[pdf_path],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[lang],
        backend="pipeline",
        device="cpu",
        # Output options - only MD
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
    )

    print(f"Conversion complete! Output: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_mineru.py <pdf_path> [output_dir] [lang]")
        print("Example: python convert_mineru.py TV-RAG.pdf ./output en")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./mineru_md_only"
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"

    convert_pdf_to_md(pdf_path, output_dir, lang)
