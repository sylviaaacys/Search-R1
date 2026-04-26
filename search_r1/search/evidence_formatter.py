import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
    "after", "before", "into", "than", "then", "there", "these", "those",
    "your", "you", "about", "can", "could", "should", "would", "will",
}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", (text or "").lower())


def _keyword_set(text: str) -> set:
    return {tok for tok in _tokenize(text) if tok not in _STOPWORDS and len(tok) > 1}


def _split_sentences(text: str) -> List[str]:
    normalized = _normalize_space(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=;)\s+", normalized)
    return [part.strip(" -") for part in parts if len(part.strip()) >= 20]


def _clip_text(text: str, max_chars: int) -> str:
    text = _normalize_space(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;:") + " ..."


def _score_text_unit(text: str, query_keywords: set) -> int:
    unit_keywords = _keyword_set(text)
    overlap = len(query_keywords & unit_keywords)
    return overlap * 10 + min(len(unit_keywords), 12)


def _summarize_sentence(text: str, query_keywords: set, target_chars: int) -> str:
    text = _normalize_space(text)
    if not text or target_chars <= 0:
        return ""
    if len(text) <= target_chars:
        return text

    # Prefer keeping the most query-relevant clauses instead of blunt truncation.
    clauses = [
        _normalize_space(part)
        for part in re.split(r"(?<=[,;:])\s+|\s+(?:and|but|while|with|showing|revealing)\s+", text)
        if _normalize_space(part)
    ]
    if len(clauses) <= 1:
        return _clip_text(text, target_chars)

    ranked = sorted(
        enumerate(clauses),
        key=lambda item: (_score_text_unit(item[1], query_keywords), -item[0]),
        reverse=True,
    )

    chosen_indices = []
    current_len = 0
    for idx, clause in ranked:
        clause_len = len(clause) + (2 if chosen_indices else 0)
        if chosen_indices and current_len + clause_len > target_chars:
            continue
        if not chosen_indices and len(clause) > target_chars:
            return _clip_text(clause, target_chars)
        chosen_indices.append(idx)
        current_len += clause_len

    if not chosen_indices:
        return _clip_text(text, target_chars)

    summary = "; ".join(clauses[idx] for idx in sorted(chosen_indices))
    if len(summary) <= target_chars:
        return summary
    return _clip_text(summary, target_chars)


def _clip_block(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if "\n" in clipped:
        lines = clipped.splitlines()
        if lines:
            lines[-1] = lines[-1].rsplit(" ", 1)[0] if " " in lines[-1] else lines[-1]
        clipped = "\n".join(lines).rstrip()
    elif " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;:") + " ..."


def _extract_doc_content(doc_item: Any) -> Tuple[str, str]:
    if isinstance(doc_item, dict):
        document = doc_item.get("document", doc_item)
        if isinstance(document, dict):
            contents = document.get("contents") or document.get("content") or ""
        else:
            contents = str(document)
    else:
        contents = str(doc_item)

    lines = [line.strip() for line in str(contents).splitlines() if line.strip()]
    if not lines:
        return "", ""
    title = lines[0].strip('"')
    body = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
    return title, body


def _extract_score(doc_item: Any) -> Optional[float]:
    if isinstance(doc_item, dict) and "score" in doc_item:
        try:
            return float(doc_item["score"])
        except (TypeError, ValueError):
            return None
    return None


def _confidence_label(scores: Sequence[Optional[float]], selected_count: int) -> str:
    numeric_scores = [score for score in scores if score is not None and not math.isnan(score)]
    if not numeric_scores:
        return "medium" if selected_count >= 2 else "low"

    max_score = max(numeric_scores)
    min_score = min(numeric_scores)
    span = abs(max_score - min_score)

    if max_score >= 20 or span >= 10:
        return "high"
    if max_score >= 5 or selected_count >= 2:
        return "medium"
    return "low"


def format_relevant_evidence(
    query: str,
    retrieval_result: Sequence[Any],
    max_total_chars: int = 1200,
    max_sentences: int = 3,
    per_sentence_chars: int = 220,
) -> str:
    query_keywords = _keyword_set(query)
    candidates: List[Dict[str, Any]] = []

    for rank, doc_item in enumerate(retrieval_result):
        title, body = _extract_doc_content(doc_item)
        score = _extract_score(doc_item)
        doc_text = body or title
        for sent_idx, sentence in enumerate(_split_sentences(doc_text)):
            sentence_keywords = _keyword_set(sentence)
            overlap = len(query_keywords & sentence_keywords)
            lexical_score = overlap * 10 + min(len(sentence_keywords), 12)
            rank_bonus = max(0, 5 - rank)
            title_bonus = 3 if title and any(tok in _keyword_set(title) for tok in query_keywords) else 0
            retrieval_bonus = 0.0 if score is None else float(score)
            total_score = lexical_score + rank_bonus + title_bonus + retrieval_bonus
            candidates.append(
                {
                    "sentence": sentence,
                    "title": title,
                    "score": total_score,
                    "retrieval_score": score,
                    "rank": rank,
                    "sent_idx": sent_idx,
                    "overlap": overlap,
                }
            )

    if not candidates:
        return _clip_block("Relevant evidence:\n1. No strong supporting evidence was retrieved.\n2. Source confidence: low.", max_total_chars)

    candidates.sort(
        key=lambda item: (
            item["overlap"],
            item["score"],
            -item["rank"],
            -item["sent_idx"],
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    seen_sentences = set()
    for item in candidates:
        normalized = item["sentence"].lower()
        if normalized in seen_sentences:
            continue
        selected.append(item)
        seen_sentences.add(normalized)
        if len(selected) >= max_sentences:
            break

    lines = ["Relevant evidence:"]
    used_scores: List[Optional[float]] = []
    for idx, item in enumerate(selected, start=1):
        sentence = _summarize_sentence(item["sentence"], query_keywords, per_sentence_chars)
        title_suffix = f" [{item['title']}]" if item["title"] else ""
        lines.append(f"{idx}. {sentence}{title_suffix}")
        used_scores.append(item["retrieval_score"])
    lines.append(f"{len(selected) + 1}. Source confidence: {_confidence_label(used_scores, len(selected))}.")

    formatted = "\n".join(lines)
    return _clip_block(formatted, max_total_chars)
