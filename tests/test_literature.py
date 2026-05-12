"""Unit tests for the OpenAlex literature client.

We avoid hitting the network by injecting an :class:`httpx.MockTransport`
so we can verify request shape + response parsing without a live API call.
"""

from __future__ import annotations

import httpx

from nanoresearch.literature import OpenAlexClient, SearchQuery
from nanoresearch.literature.client import _OPENALEX_URL  # type: ignore[attr-defined]


def _fake_openalex_response() -> dict[str, object]:
    # Abstract: "We propose a method that achieves 92.4% accuracy and F1 of 0.81 on PubMedQA."
    inv = {
        "We": [0],
        "propose": [1],
        "a": [2],
        "method": [3],
        "that": [4],
        "achieves": [5],
        "92.4%": [6],
        "accuracy": [7],
        "and": [8],
        "F1": [9],
        "of": [10],
        "0.81": [11],
        "on": [12],
        "PubMedQA.": [13],
    }
    return {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "display_name": "Tiny BiomedQA Improvements",
                "abstract_inverted_index": inv,
                "publication_year": 2024,
                "doi": "https://doi.org/10.0000/example",
                "authorships": [
                    {"author": {"display_name": "Alice X", "orcid": None}},
                    {"author": {"display_name": "Bob Y", "orcid": None}},
                ],
                "cited_by_count": 42,
                "primary_location": {
                    "source": {"display_name": "EMNLP"},
                    "pdf_url": "https://example.com/paper.pdf",
                },
            }
        ]
    }


def test_openalex_search_parses_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert _OPENALEX_URL in str(request.url)
        assert "search=biomedical+qa" in str(request.url) or "search=biomedical%20qa" in str(request.url)
        return httpx.Response(200, json=_fake_openalex_response())

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = OpenAlexClient(client=http)

    papers = client.search(SearchQuery(text="biomedical qa", max_results=5))
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Tiny BiomedQA Improvements"
    assert p.year == 2024
    assert p.citations == 42
    assert p.venue == "EMNLP"
    assert len(p.authors) == 2
    assert "92.4%" in p.abstract
    assert p.url == "https://example.com/paper.pdf"


def test_openalex_search_caches_identical_queries() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=_fake_openalex_response())

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAlexClient(client=http)
    q = SearchQuery(text="biomedical qa", max_results=5)
    client.search(q)
    client.search(q)
    client.search(q)
    assert call_count == 1


def test_evidence_extraction_finds_metrics() -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_fake_openalex_response())))
    client = OpenAlexClient(client=http)
    papers = client.search(SearchQuery(text="biomedical qa", max_results=1))
    evs = client.extract_evidence(papers[0])
    metrics = {e.metric for e in evs}
    # Accuracy percent and F1 should both appear; generic percent will also match the 92.4%.
    assert "accuracy_percent" in metrics
    assert "f1" in metrics


def test_openalex_search_retries_on_http_error() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_fake_openalex_response())

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAlexClient(client=http, max_retries=4)
    papers = client.search(SearchQuery(text="biomedical qa", max_results=1))
    assert attempts == 3
    assert len(papers) == 1
