"""Literature data models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str
    orcid: str | None = None


class Evidence(BaseModel):
    """One *quantitative* extract from a paper (e.g., 'Acc 92.4% on PubMedQA').

    The ideation phase parses these from abstracts/PDFs so that hypotheses
    can be grounded in real numbers rather than the LLM's prior (paper §3.2.1).
    """

    metric: str
    value: float | str
    dataset: str = ""
    method: str = ""
    snippet: str = ""


class Paper(BaseModel):
    paper_id: str = Field(..., description="Source-stable identifier (e.g., openalex id, arxiv id).")
    source: str = Field(..., description="Backing source: 'openalex' | 'arxiv' | …")
    title: str
    abstract: str = ""
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    citations: int = 0
    doi: str | None = None
    url: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def short_citation(self) -> str:
        first = self.authors[0].name if self.authors else "Unknown"
        yr = self.year or "n.d."
        return f"{first} et al. ({yr}) — {self.title}"


class SearchQuery(BaseModel):
    text: str
    max_results: int = 10
    year_from: int | None = None
    year_to: int | None = None
    sort: str = "relevance_score:desc"
