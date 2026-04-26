from typing import Dict, List, Tuple


def _safe_metadata_value(metadata: Dict, key: str, default: str = "") -> str:
    value = metadata.get(key, default)
    if value is None:
        return default
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value)


def fetch_pubmed_documents(query: str, topk: int) -> Tuple[List[Dict[str, str]], List[float]]:
    from langchain_community.document_loaders import PubMedLoader
    loader = PubMedLoader(query=query, load_max_docs=topk)
    docs = loader.load()

    results: List[Dict[str, str]] = []
    scores: List[float] = []

    for rank, doc in enumerate(docs):
        metadata = getattr(doc, "metadata", {}) or {}
        title = _safe_metadata_value(metadata, "Title")
        published = _safe_metadata_value(metadata, "Published")
        summary = (getattr(doc, "page_content", "") or "").strip()
        link = _safe_metadata_value(metadata, "link")
        pubmed_id = _safe_metadata_value(metadata, "uid") or _safe_metadata_value(metadata, "pmid")

        parts = []
        if title:
            parts.append(title)
        if published:
            parts.append(f"Published: {published}")
        if summary:
            parts.append(summary)
        if link:
            parts.append(f"Link: {link}")

        results.append(
            {
                "id": pubmed_id or f"pubmed-{rank}",
                "title": title,
                "texts": summary,
                "contents": "\n".join(parts).strip(),
            }
        )
        scores.append(float(topk - rank))

    return results, scores

print(fetch_pubmed_documents("lichen planus pigmentosus violaceous brown macules Civatte bodies melanin incontinence",3))