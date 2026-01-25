# PDF to Markdown 변환기 설치 및 사용 가이드

PDF를 Markdown/Text로 변환하는 도구들의 설치 방법 및 사용법

## 목차

- [비교 요약](#비교-요약)
- [1. pdftotext (Poppler)](#1-pdftotext-poppler)
- [2. MarkItDown](#2-markitdown)
- [3. PyMuPDF4LLM (권장)](#3-pymupdf4llm-권장)
- [4. MinerU](#4-mineru)
- [빠른 시작](#빠른-시작)
- [선택 가이드](#선택-가이드)

---

## 비교 요약

| 항목 | pdftotext | MarkItDown | PyMuPDF4LLM | MinerU |
|------|-----------|------------|-------------|--------|
| **개발사** | Poppler | Microsoft | Artifex | OpenDataLab |
| **출력** | Plain Text | Markdown | Markdown | Markdown |
| **속도** | 가장 빠름 | 빠름 | 빠름 (~1초) | 느림 (~3분) |
| **수식** | 없음 | 없음 | 이탤릭 | LaTeX |
| **표** | 텍스트 | 텍스트 | 텍스트 | HTML |
| **이미지 추출** | 없음 | 없음 | PNG | PNG |
| **레이아웃 유지** | 옵션 | 제한적 | 좋음 | 매우 좋음 |
| **설치 복잡도** | 시스템 패키지 | 간단 | 간단 | 복잡 |
| **의존성** | 없음 | 적음 | 적음 | 많음 (PyTorch) |
| **추천 용도** | 텍스트 추출 | 일반 문서 | RAG, 일반 | 학술 논문 |

---

## 1. pdftotext (Poppler)

Poppler 라이브러리 기반의 시스템 레벨 PDF 텍스트 추출 도구

### 설치

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
```bash
# Chocolatey
choco install poppler

# 또는 수동 설치: https://github.com/oschwartz10612/poppler-windows/releases
```

### 사용법

**CLI:**
```bash
# 기본 변환 (텍스트 파일 생성)
pdftotext document.pdf output.txt

# 레이아웃 유지
pdftotext -layout document.pdf output.txt

# 표 구조 유지 (고정폭)
pdftotext -table document.pdf output.txt

# 특정 페이지만 (1-based)
pdftotext -f 1 -l 5 document.pdf output.txt

# 표준 출력으로
pdftotext document.pdf -

# 인코딩 지정
pdftotext -enc UTF-8 document.pdf output.txt
```

**주요 옵션:**
| 옵션 | 설명 |
|------|------|
| `-layout` | 원본 레이아웃 유지 |
| `-table` | 표 구조 유지 (고정폭 폰트용) |
| `-raw` | 콘텐츠 스트림 순서대로 출력 |
| `-f <int>` | 시작 페이지 (1-based) |
| `-l <int>` | 끝 페이지 (1-based) |
| `-enc <string>` | 출력 인코딩 (UTF-8, Latin1 등) |
| `-nopgbrk` | 페이지 구분자 제거 |
| `-eol <string>` | 줄바꿈 문자 (unix, dos, mac) |

### 장점
- 가장 빠른 변환 속도
- 시스템 레벨 도구 (Python 불필요)
- 안정적이고 검증된 도구
- 레이아웃 옵션 제공

### 단점
- Plain Text만 출력 (Markdown 아님)
- 이미지 추출 불가
- 수식/표 서식 없음

---

## 2. MarkItDown

Microsoft에서 개발한 경량 문서 변환기

### 설치

```bash
# 기본 설치
pip install markitdown

# 모든 기능 포함
pip install markitdown[all]

# uv 사용
uv pip install markitdown[all]
```

### 사용법

**CLI:**
```bash
# 기본 변환
markitdown document.pdf > output.md

# 출력 파일 지정
markitdown document.pdf -o output.md

# 파이프 사용
cat document.pdf | markitdown
```

**Python:**
```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("document.pdf")
print(result.text_content)
```

### 장점
- 설치가 간단
- 다양한 포맷 지원 (PDF, DOCX, PPTX, XLSX, HTML 등)
- 가벼운 의존성

### 단점
- 수식 지원 없음
- 이미지 추출 없음
- 복잡한 레이아웃 처리 제한적

---

## 3. PyMuPDF4LLM (권장)

PyMuPDF 기반의 LLM/RAG 최적화 변환기

### 설치

```bash
# 기본 설치
pip install pymupdf4llm

# uv 사용
uv pip install pymupdf4llm
```

### 사용법

**스크립트 (convert_pymupdf.py):**
```bash
# 기본 변환
python convert_pymupdf.py document.pdf

# 출력 경로 지정
python convert_pymupdf.py document.pdf -o output.md

# 특정 페이지만 (0-based index)
python convert_pymupdf.py document.pdf --pages 0 1 2

# 이미지 없이
python convert_pymupdf.py document.pdf --no-images

# 고해상도 이미지 (기본 150 DPI)
python convert_pymupdf.py document.pdf --dpi 300

# 이미지 저장 경로 지정
python convert_pymupdf.py document.pdf --image-path ./images
```

**Python:**
```python
import pymupdf4llm
import pathlib

# 기본 변환
md_text = pymupdf4llm.to_markdown("document.pdf")
pathlib.Path("output.md").write_bytes(md_text.encode())

# 이미지 추출 포함
md_text = pymupdf4llm.to_markdown(
    "document.pdf",
    write_images=True,
    image_path="./images",
    dpi=150
)

# 특정 페이지만
md_text = pymupdf4llm.to_markdown(
    "document.pdf",
    pages=[0, 1, 2]  # 0-based
)

# 페이지별 청크 반환 (RAG용)
chunks = pymupdf4llm.to_markdown(
    "document.pdf",
    page_chunks=True,
    extract_words=True
)
for chunk in chunks:
    print(f"Page {chunk['metadata']['page']}")
    print(chunk['text'][:500])
```

### 장점
- 빠른 변환 속도
- 이미지 자동 추출
- Markdown 서식 자동 적용 (볼드, 이탤릭, 링크)
- RAG/LLM에 최적화된 출력
- 페이지별 청크 지원

### 단점
- 수식을 LaTeX로 변환하지 않음
- 표를 HTML로 변환하지 않음

---

## 4. MinerU

OpenDataLab에서 개발한 고급 PDF 추출기 (레이아웃 분석 + OCR)

### 설치

```bash
# 기본 설치
pip install mineru

# uv 사용
uv pip install mineru

# 필수 의존성 (자동 설치 안 될 수 있음)
uv pip install torch==2.5.1 torchvision==0.20.1
uv pip install ultralytics doclayout-yolo
uv pip install transformers tokenizers
uv pip install omegaconf shapely pyclipper dill ftfy
```

### Apple Silicon (M1/M2/M3) 주의사항

MPS(Metal) 제한으로 인해 CPU 모드 강제 필요:

```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
```

### 사용법

**CLI:**
```bash
# 기본 변환 (모든 출력 파일 생성)
mineru -p document.pdf -o ./output -b pipeline -l en -d cpu

# 한국어 문서
mineru -p document.pdf -o ./output -b pipeline -l korean -d cpu
```

**스크립트 (convert_mineru.py) - MD만 출력:**
```bash
# 기본 변환
python convert_mineru.py document.pdf

# 출력 디렉토리 지정
python convert_mineru.py document.pdf ./output

# 언어 지정 (en, ch, korean, japan 등)
python convert_mineru.py document.pdf ./output korean
```

**Python:**
```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from mineru.cli.common import do_parse
from pathlib import Path

pdf_bytes = Path("document.pdf").read_bytes()

do_parse(
    output_dir="./output",
    pdf_file_names=["document.pdf"],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=["en"],
    backend="pipeline",
    device="cpu",
    # MD만 출력
    f_dump_md=True,
    f_dump_middle_json=False,
    f_dump_model_output=False,
    f_dump_orig_pdf=False,
    f_dump_content_list=False,
    f_draw_layout_bbox=False,
    f_draw_span_bbox=False,
)
```

### 지원 언어

| 코드 | 언어 |
|------|------|
| `en` | 영어 |
| `ch` | 중국어 (간체) |
| `chinese_cht` | 중국어 (번체) |
| `korean` | 한국어 |
| `japan` | 일본어 |
| `latin` | 라틴어 |
| `arabic` | 아랍어 |

### 장점
- 수식을 LaTeX로 변환
- 표를 HTML로 변환
- 정확한 레이아웃 분석
- OCR 내장

### 단점
- 설치가 복잡 (많은 의존성)
- 변환 속도가 느림 (~3분/문서)
- 큰 용량 (PyTorch + 모델 파일)
- Apple Silicon에서 추가 설정 필요

---

## 빠른 시작

### 전체 설치

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 의존성 설치
uv pip install -r requirements.txt
```

### 변환 비교 테스트

```bash
# pdftotext (가장 빠름, Plain Text)
pdftotext -layout document.pdf document_pdftotext.txt

# MarkItDown
markitdown document.pdf -o document_markitdown.md

# PyMuPDF4LLM (권장)
python convert_pymupdf.py document.pdf -o document_pymupdf.md

# MinerU
python convert_mineru.py document.pdf ./mineru_output en
```

---

## 선택 가이드

| 상황 | 추천 도구 |
|------|-----------|
| 단순 텍스트 추출 | pdftotext |
| Python 없이 사용 | pdftotext |
| 빠른 변환이 필요할 때 | pdftotext, PyMuPDF4LLM |
| RAG 파이프라인용 | PyMuPDF4LLM |
| Markdown 서식 필요 | PyMuPDF4LLM, MarkItDown |
| 수식이 많은 학술 논문 | MinerU |
| 표가 복잡한 문서 | MinerU |
| 간단한 문서 변환 | MarkItDown |
| 설치가 간편해야 할 때 | pdftotext, MarkItDown |
| 정확한 레이아웃이 필요할 때 | MinerU |
| 이미지 추출 필요 | PyMuPDF4LLM, MinerU |

---

## 참고 링크

- [Poppler (pdftotext)](https://poppler.freedesktop.org/)
- [MarkItDown](https://github.com/microsoft/markitdown)
- [PyMuPDF4LLM](https://github.com/pymupdf/pymupdf4llm)
- [MinerU](https://github.com/opendatalab/MinerU)
