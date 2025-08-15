"""Test configuration and utilities."""

import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()
        (data_dir / "raw").mkdir()
        (data_dir / "processed").mkdir()
        (data_dir / "graphs").mkdir()
        yield str(data_dir)


@pytest.fixture
def sample_generifs():
    """Sample GeneRIF data for testing."""
    return [
        {
            'tax_id': 9606,
            'gene_id': 1,
            'pmid': 12345,
            'timestamp': '2023-01-01',
            'text': 'The TP53 gene is associated with cancer development and tumor suppression.'
        },
        {
            'tax_id': 9606,
            'gene_id': 2,
            'pmid': 12346,
            'timestamp': '2023-01-02',
            'text': 'BRCA1 mutations increase risk of breast cancer through DNA repair defects.'
        },
        {
            'tax_id': 9606,
            'gene_id': 3,
            'pmid': 12347,
            'timestamp': '2023-01-03',
            'text': 'EGFR overexpression promotes cell proliferation in lung cancer patients.'
        }
    ]


@pytest.fixture
def sample_relations():
    """Sample relation triplets for testing."""
    return [
        {
            'subject': {'text': 'TP53', 'label': 'GENE'},
            'predicate': 'associated_with',
            'object': {'text': 'cancer', 'label': 'DISEASE'},
            'confidence': 0.9
        },
        {
            'subject': {'text': 'BRCA1', 'label': 'GENE'},
            'predicate': 'causes',
            'object': {'text': 'breast cancer', 'label': 'DISEASE'},
            'confidence': 0.8
        },
        {
            'subject': {'text': 'EGFR', 'label': 'GENE'},
            'predicate': 'promotes',
            'object': {'text': 'cell proliferation', 'label': 'FUNCTION'},
            'confidence': 0.7
        }
    ]


# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_GENERIF_FILE = TEST_DATA_DIR / "sample_generifs.csv"
SAMPLE_RELATIONS_FILE = TEST_DATA_DIR / "sample_relations.pkl"
