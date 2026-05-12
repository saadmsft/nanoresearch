"""Academic literature retrieval.

Currently uses **OpenAlex** as the primary source — free, no API key required,
covers ~250M works (including arXiv, conference papers, journals). arXiv and
Semantic Scholar adapters can be plugged in later via the same
:class:`LiteratureClient` protocol.
"""

from .client import LiteratureClient, OpenAlexClient
from .models import Author, Evidence, Paper, SearchQuery

__all__ = [
    "Author",
    "Evidence",
    "LiteratureClient",
    "OpenAlexClient",
    "Paper",
    "SearchQuery",
]
