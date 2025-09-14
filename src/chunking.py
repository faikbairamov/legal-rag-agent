import re
from typing import List, Dict, Optional, Tuple, Iterable, Iterator


def _find_article_sections(text: str) -> List[Tuple[int, int, Dict[str, str]]]:
    """
    Find sections by Georgian legal article headers ("მუხლი N.").

    Returns list of (start, end, meta) for each article block.
    If nothing matches, returns single block covering all text.
    """
    # Match lines like: "მუხლი 12." or "მუხლი 12 " or " მუხლი 12. ..."
    article_re = re.compile(r"(?m)^(\s*მუხლი\s+(\d+)[\.|\s].*)$")
    matches = list(article_re.finditer(text))
    if not matches:
        return [(0, len(text), {"section_title": "FULL_TEXT", "article": ""})]

    sections: List[Tuple[int, int, Dict[str, str]]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title_line = m.group(1).strip()
        article_no = m.group(2)
        sections.append((start, end, {"section_title": title_line, "article": article_no}))
    return sections


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token; avoid heavy deps here.
    return max(1, len(text) // 4)


def iter_chunks(
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
    use_sections: bool = True,
) -> Iterator[Dict]:
    """Yield chunk dicts without holding all in memory."""
    blocks: List[Tuple[int, int, Dict[str, str]]]
    if use_sections:
        blocks = _find_article_sections(text)
    else:
        blocks = [(0, len(text), {"section_title": "FULL_TEXT", "article": ""})]

    for (b_start, b_end, meta) in blocks:
        block = text[b_start:b_end]
        start = 0
        while start < len(block):
            approx_target_chars = target_tokens * 4
            end = min(len(block), start + approx_target_chars)

            boundary = -1
            for sep in ["\n\n", "\n", ". ", "? ", "! "]:
                idx = block.rfind(sep, start, end)
                if idx != -1:
                    boundary = idx + len(sep)
                    break
            if boundary != -1 and boundary > start:
                end = boundary

            content = block[start:end].strip()
            if not content:
                break
            abs_start = b_start + start
            abs_end = b_start + end
            yield {
                "content": content,
                "start": abs_start,
                "end": abs_end,
                "section_title": meta.get("section_title", ""),
                "article": meta.get("article", ""),
            }

            if end >= len(block):
                break
            overlap_chars = max(0, overlap_tokens * 4)
            start = max(0, end - overlap_chars)


def split_into_chunks(
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
    use_sections: bool = True,
) -> List[Dict]:
    """Backwards-compatible wrapper that collects all chunks into a list."""
    return list(
        iter_chunks(
            text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            use_sections=use_sections,
        )
    )
