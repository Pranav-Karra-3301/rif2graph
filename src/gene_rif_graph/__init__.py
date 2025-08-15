"""
Gene-RIF Graph: Convert GeneRIFs to knowledge graphs using biomedical NLP.

This package provides tools for:
1. Downloading and parsing GeneRIF data from NCBI
2. Extracting biomedical entities and relations using NLP
3. Building and analyzing knowledge graphs
4. Computing graph statistics and community detection
"""

__version__ = "0.1.0"
__author__ = "Pranav Karra"
__email__ = "your_email@example.com"

from .ingest import GeneRIFDownloader, PubMedFetcher
from .nlp import BioNERExtractor, RelationExtractor
from .filters import TripleFilter
from .normalize import GeneNormalizer, ConceptNormalizer
from .graph import GraphBuilder, GraphAnalyzer

__all__ = [
    "GeneRIFDownloader",
    "PubMedFetcher", 
    "BioNERExtractor",
    "RelationExtractor",
    "TripleFilter",
    "GeneNormalizer",
    "ConceptNormalizer",
    "GraphBuilder",
    "GraphAnalyzer",
]
