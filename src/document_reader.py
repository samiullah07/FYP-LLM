# src/document_reader.py
"""
Document Reader — extracts text from various input formats.

Supports:
    - Plain text string (already works)
    - .txt files
    - .pdf files
    - .docx files
    - Multi-paragraph research briefs
    - CSV files with multiple topics

Output is always a clean topic string or list of topic strings
that can be fed directly into the pipeline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Read plain text file
# ---------------------------------------------------------------------------
def read_txt_file(path: str | Path) -> str:
    """
    Read a plain text file and return its content as a string.

    Parameters
    ----------
    path : str or Path
        Path to the .txt file.

    Returns
    -------
    str : full text content
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"[DocumentReader] Read TXT: {len(text)} chars from {path.name}")
    return text


# ---------------------------------------------------------------------------
# Read PDF file
# ---------------------------------------------------------------------------
def read_pdf_file(path: str | Path) -> str:
    """
    Extract text from a PDF file.

    Reads all pages and joins them into a single string.

    Parameters
    ----------
    path : str or Path
        Path to the .pdf file.

    Returns
    -------
    str : extracted text from all pages
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("Install PyPDF2: pip install PyPDF2")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text_parts = []

    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total_pages = len(reader.pages)
        print(f"[DocumentReader] Reading PDF: {total_pages} pages from {path.name}")

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

    full_text = "\n\n".join(text_parts)
    print(f"[DocumentReader] Extracted {len(full_text)} chars from PDF")
    return full_text


# ---------------------------------------------------------------------------
# Read Word document
# ---------------------------------------------------------------------------
def read_docx_file(path: str | Path) -> str:
    """
    Extract text from a Word .docx file.

    Parameters
    ----------
    path : str or Path
        Path to the .docx file.

    Returns
    -------
    str : extracted text from all paragraphs
    """
    try:
        import docx
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    doc   = docx.Document(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paras)

    print(f"[DocumentReader] Read DOCX: {len(paras)} paragraphs, "
          f"{len(full_text)} chars from {path.name}")
    return full_text


# ---------------------------------------------------------------------------
# Read CSV of topics
# ---------------------------------------------------------------------------
def read_topics_from_csv(path: str | Path, column: str = "topic") -> list[str]:
    """
    Read multiple topics from a CSV file.

    Parameters
    ----------
    path   : str or Path
        Path to the .csv file.
    column : str
        Column name containing topics (default: "topic").

    Returns
    -------
    list[str] : list of topic strings
    """
    import csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    topics = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column in row and row[column].strip():
                topics.append(row[column].strip())

    print(f"[DocumentReader] Loaded {len(topics)} topics from {path.name}")
    return topics


# ---------------------------------------------------------------------------
# Summarise a long document into a focused topic string
# ---------------------------------------------------------------------------
def summarise_document_to_topic(
    text: str,
    max_chars: int = 3000,
) -> str:
    """
    If a document is very long, extract the most relevant portion
    to use as the topic context for the pipeline.

    Strategy:
        1. If text <= max_chars: use as-is
        2. If text > max_chars: take first 1500 + last 1500 chars
           (captures abstract/intro + conclusion)

    Parameters
    ----------
    text     : str   full document text
    max_chars: int   maximum characters to keep

    Returns
    -------
    str : trimmed topic context
    """
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    trimmed = (
        text[:half]
        + "\n\n[...document truncated for processing...]\n\n"
        + text[-half:]
    )
    print(f"[DocumentReader] Document trimmed: {len(text)} → {len(trimmed)} chars")
    return trimmed


# ---------------------------------------------------------------------------
# Extract topic from long document using LLM
# ---------------------------------------------------------------------------
def extract_topic_from_document(text: str) -> str:
    """
    Use Groq LLM to extract a focused research topic from a long document.

    This is the smart approach:
        1. Send the first 3000 chars of the document to the LLM
        2. Ask it to extract the core research topic as a single sentence
        3. Use that sentence as the topic for the pipeline

    Parameters
    ----------
    text : str
        Full document text (e.g., from PDF or DOCX).

    Returns
    -------
    str : focused topic string (1-3 sentences max)
    """
    from groq import Groq
    from src.config import settings

    client = Groq(api_key=settings.groq_api_key)

    # Use first 3000 chars to stay within token limits
    excerpt = text[:3000]

    prompt = (
        "You are a research assistant. Read the following document excerpt "
        "and extract the core research topic in 1-2 sentences.\n\n"
        "Rules:\n"
        "- Be specific and technical\n"
        "- Include key domain terms\n"
        "- Maximum 50 words\n"
        "- Return ONLY the topic statement, nothing else\n\n"
        f"Document excerpt:\n{excerpt}"
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.1,
    )

    topic = response.choices[0].message.content.strip()
    print(f"[DocumentReader] Extracted topic: '{topic}'")
    return topic


# ---------------------------------------------------------------------------
# Master function — accepts any input type
# ---------------------------------------------------------------------------
def load_input(
    source: str | Path,
    mode:   str = "auto",
) -> str:
    """
    Universal input loader — accepts any supported format.

    Parameters
    ----------
    source : str or Path
        - A topic string (plain text, not a file path)
        - Path to .txt, .pdf, or .docx file

    mode : str
        - "auto"   : detect format automatically (default)
        - "topic"  : treat source as a plain topic string
        - "txt"    : read as text file
        - "pdf"    : read as PDF
        - "docx"   : read as Word document
        - "extract": read file and extract topic using LLM

    Returns
    -------
    str : topic or document text ready for pipeline
    """
    source = str(source)

    # Auto-detect mode
    if mode == "auto":
        if source.endswith(".pdf"):
            mode = "pdf"
        elif source.endswith(".docx"):
            mode = "docx"
        elif source.endswith(".txt"):
            mode = "txt"
        else:
            mode = "topic"

    if mode == "topic":
        print(f"[DocumentReader] Using plain topic string ({len(source)} chars)")
        return source

    elif mode == "txt":
        return read_txt_file(source)

    elif mode == "pdf":
        text = read_pdf_file(source)
        return summarise_document_to_topic(text)

    elif mode == "docx":
        text = read_docx_file(source)
        return summarise_document_to_topic(text)

    elif mode == "extract":
        # Read file then use LLM to extract topic
        if source.endswith(".pdf"):
            text = read_pdf_file(source)
        elif source.endswith(".docx"):
            text = read_docx_file(source)
        elif source.endswith(".txt"):
            text = read_txt_file(source)
        else:
            text = source
        return extract_topic_from_document(text)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'topic', 'txt', 'pdf', 'docx', or 'extract'")