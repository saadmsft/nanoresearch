"""HTTP clients for academic databases.

Currently :class:`OpenAlexClient` is the only implementation. It exposes
search + evidence extraction. Calls are cached in-memory per process to
avoid re-fetching during a single run.
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

import httpx

from ..logging import get_logger
from .models import Author, Evidence, Paper, SearchQuery

_log = get_logger(__name__)

# OpenAlex politeness: include an email in the User-Agent for higher rate-limits.
_USER_AGENT = "NanoResearch/0.1 (mailto:research-bot@example.com)"
_OPENALEX_URL = "https://api.openalex.org/works"


class LiteratureClient(Protocol):
    """Common interface for any literature backend."""

    def search(self, query: SearchQuery) -> list[Paper]: ...
    def extract_evidence(self, paper: Paper) -> list[Evidence]: ...


# =========================================================================
# OpenAlex
# =========================================================================


class OpenAlexClient:
    """OpenAlex REST client. No API key required."""

    name = "openalex"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
        max_retries: int = 3,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        )
        self._max_retries = max_retries
        self._cache: dict[str, list[Paper]] = {}

    # ------------------------------------------------------------- search

    def search(self, query: SearchQuery) -> list[Paper]:
        cache_key = self._cache_key(query)
        if cache_key in self._cache:
            return self._cache[cache_key]

        params: dict[str, Any] = {
            "search": query.text,
            "per-page": min(max(query.max_results, 1), 25),
            "sort": query.sort,
            "select": (
                "id,display_name,abstract_inverted_index,publication_year,"
                "doi,authorships,cited_by_count,primary_location"
            ),
        }
        filters: list[str] = []
        if query.year_from:
            filters.append(f"from_publication_date:{query.year_from}-01-01")
        if query.year_to:
            filters.append(f"to_publication_date:{query.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        data = self._get(_OPENALEX_URL, params)
        works = data.get("results", []) or []
        papers = [self._parse_work(w) for w in works]
        self._cache[cache_key] = papers
        _log.info(
            "openalex_search",
            query=query.text,
            n_results=len(papers),
            max_results=query.max_results,
        )
        return papers

    def _parse_work(self, work: dict[str, Any]) -> Paper:
        title = work.get("display_name") or ""
        authors = [
            Author(
                name=a.get("author", {}).get("display_name", ""),
                orcid=a.get("author", {}).get("orcid"),
            )
            for a in (work.get("authorships") or [])
            if a.get("author", {}).get("display_name")
        ]
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))
        primary = work.get("primary_location") or {}
        source_info = (primary.get("source") or {}) if isinstance(primary, dict) else {}
        venue = source_info.get("display_name", "") if isinstance(source_info, dict) else ""
        url = (
            (primary.get("pdf_url") if isinstance(primary, dict) else None)
            or work.get("doi")
            or work.get("id")
        )
        return Paper(
            paper_id=str(work.get("id") or ""),
            source=self.name,
            title=title,
            abstract=abstract,
            authors=authors,
            year=work.get("publication_year"),
            venue=venue,
            citations=int(work.get("cited_by_count") or 0),
            doi=work.get("doi"),
            url=url,
            raw=work,
        )

    @staticmethod
    def _reconstruct_abstract(inv: dict[str, list[int]] | None) -> str:
        """OpenAlex stores abstracts as inverted indices to dodge copyright."""
        if not inv:
            return ""
        flat: list[tuple[int, str]] = []
        for word, positions in inv.items():
            for p in positions:
                flat.append((p, word))
        flat.sort(key=lambda x: x[0])
        return " ".join(w for _, w in flat)

    # ------------------------------------------------------------- evidence

    # Common quantitative patterns: "92.4%", "F1 of 0.81", "accuracy = 0.87",
    # "BLEU 32.1", "ROUGE-L 41.2".
    _METRIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        # "92.4% accuracy" / "92.4 % accuracy"
        (
            "accuracy_percent",
            re.compile(r"([\d.]+)\s*%\s*(?:accuracy|acc)\b", re.I),
        ),
        # "accuracy of 92.4%" / "accuracy = 0.924" / "accuracy: 92.4"
        (
            "accuracy_percent",
            re.compile(r"(?:accuracy|acc(?:uracy)?)\s*(?:of|=|:)?\s*([\d.]+)\s*%?", re.I),
        ),
        ("f1", re.compile(r"\bf1\b\s*(?:score|of|=|:)?\s*([\d.]+)", re.I)),
        ("bleu", re.compile(r"\bbleu(?:-\d)?\b\s*(?:score|of|=|:)?\s*([\d.]+)", re.I)),
        ("rouge", re.compile(r"\brouge(?:-\w+)?\b\s*(?:score|of|=|:)?\s*([\d.]+)", re.I)),
        ("emr", re.compile(r"\bem(?:r)?\b\s*(?:score|of|=|:)?\s*([\d.]+)", re.I)),
        ("auroc", re.compile(r"\bauroc?\b\s*(?:of|=|:)?\s*([\d.]+)", re.I)),
        ("percent_generic", re.compile(r"([\d.]+)\s*%\b")),
    )

    def extract_evidence(self, paper: Paper) -> list[Evidence]:
        """Best-effort quantitative extraction from a paper's abstract.

        Paper §3.2.1 calls this the **quantitative evidence extraction
        mechanism**; we keep it cheap (regex over the abstract) and let the
        ideation LLM filter / reinterpret the matches.
        """
        text = paper.abstract or ""
        out: list[Evidence] = []
        for metric, pat in self._METRIC_PATTERNS:
            for m in pat.finditer(text):
                snippet = self._context_snippet(text, m.start(), m.end())
                try:
                    value: float | str = float(m.group(1))
                except (ValueError, IndexError):
                    value = m.group(0)
                out.append(
                    Evidence(
                        metric=metric,
                        value=value,
                        snippet=snippet,
                        method=paper.title[:120],
                    )
                )
        return out[:20]  # cap to keep prompts small

    @staticmethod
    def _context_snippet(text: str, start: int, end: int, *, radius: int = 60) -> str:
        a = max(0, start - radius)
        b = min(len(text), end + radius)
        snippet = text[a:b].strip()
        if a > 0:
            snippet = "…" + snippet
        if b < len(text):
            snippet = snippet + "…"
        return snippet

    # ------------------------------------------------------------- internals

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as e:
                last_exc = e
                wait = 0.5 * (2 ** attempt)
                _log.warning("openalex_retry", attempt=attempt + 1, wait_s=wait, error=str(e))
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _cache_key(q: SearchQuery) -> str:
        return f"{q.text}|{q.max_results}|{q.year_from}|{q.year_to}|{q.sort}"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
