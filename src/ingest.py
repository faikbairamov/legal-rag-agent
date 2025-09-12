import re
from pathlib import Path
import pymupdf as fitz  # type: ignore


# ეს გამოვიყენე ყოველი გვერდის ბოლოს არსებული "http://www.matsne.gov.ge" და 
# ბარკოდების, მაგ: "40.000.000.05.001.000.22" მოსაშორებლად.
URL_RE = re.compile(r"^\s*(?:https?://)?(?:www\.)?matsne\.gov\.ge/?\s*$", re.I)
CODE_RE = re.compile(r"^\s*\d{2,3}(?:\.\d{2,3}){5,}\s*$")


def clean_lines(text):
    """Remove known repeating header/footer lines and tidy whitespace."""
    cleaned = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if URL_RE.match(s):
            continue
        if CODE_RE.match(s):
            continue
        cleaned.append(s)

    # collapse excessive empty lines
    out = []
    prev_blank = False
    for l in cleaned:
        if l == "":
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(l)
            prev_blank = False
    return "\n".join(out).strip()


def extract_with_pymupdf(pdf_path: Path, header_margin: float = 80.0, footer_margin: float = 80.0) -> str:
    doc = fitz.open(pdf_path)  # type: ignore
    pages_text = []
    for page in doc:
        height = page.rect.height
        # Extract blocks and filter out header/footer by geometry
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        parts = []
        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, x1, y1, text = b[:5]
            # Skip header/footer regions
            if y1 <= header_margin:
                continue
            if y0 >= (height - footer_margin):
                continue
            if not text:
                continue
            parts.append(text)
        page_text = "\n".join(parts)
        page_text = clean_lines(page_text)
        if page_text:
            pages_text.append(page_text)
    return "\n\n".join(pages_text)

def ingest(
    input_pdf = "./data/raw/matsne-31702-134.pdf",
    output_txt = "./data/processed/matsne-31702-134.txt",
    header_margin = 80.0,
    footer_margin = 80.0,
):
    in_path = Path(input_pdf)
    out_path = Path(output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fitz is not None:
        text = extract_with_pymupdf(in_path, header_margin=header_margin, footer_margin=footer_margin)

    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    ingest()
