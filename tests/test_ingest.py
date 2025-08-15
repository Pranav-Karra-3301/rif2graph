"""Test data ingestion functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from gene_rif_graph.ingest import GeneRIFDownloader, PubMedFetcher


class TestGeneRIFDownloader:
    """Test GeneRIF data downloading and parsing."""
    
    def test_init(self, temp_data_dir):
        """Test downloader initialization."""
        downloader = GeneRIFDownloader(data_dir=temp_data_dir)
        assert downloader.data_dir.exists()
        assert downloader.data_dir.name == "data"
    
    def test_parse_generifs(self, temp_data_dir, sample_generifs):
        """Test GeneRIF parsing functionality."""
        downloader = GeneRIFDownloader(data_dir=temp_data_dir)
        
        # Create sample data file
        import csv
        sample_file = downloader.data_dir / "test_generifs.txt"
        
        with open(sample_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['#tax_id', 'gene_id', 'pmid', 'timestamp', 'text'])
            for generif in sample_generifs:
                writer.writerow([
                    generif['tax_id'], generif['gene_id'], generif['pmid'],
                    generif['timestamp'], generif['text']
                ])
        
        # Parse the file
        df = downloader.parse_generifs(sample_file)
        
        assert len(df) == len(sample_generifs)
        assert 'gene_id' in df.columns
        assert 'text' in df.columns
        assert df['gene_id'].notna().all()
    
    def test_species_filter(self, temp_data_dir, sample_generifs):
        """Test species filtering functionality."""
        downloader = GeneRIFDownloader(data_dir=temp_data_dir)
        
        # Add non-human data
        sample_generifs.append({
            'tax_id': 10090,  # Mouse
            'gene_id': 4,
            'pmid': 12348,
            'timestamp': '2023-01-04',
            'text': 'Mouse gene example.'
        })
        
        # Create sample data file
        import csv
        sample_file = downloader.data_dir / "test_generifs.txt"
        
        with open(sample_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['#tax_id', 'gene_id', 'pmid', 'timestamp', 'text'])
            for generif in sample_generifs:
                writer.writerow([
                    generif['tax_id'], generif['gene_id'], generif['pmid'],
                    generif['timestamp'], generif['text']
                ])
        
        # Parse with human filter
        df = downloader.parse_generifs(sample_file, species_filter=[9606])
        
        assert len(df) == 3  # Only human GeneRIFs
        assert (df['tax_id'] == 9606).all()


class TestPubMedFetcher:
    """Test PubMed abstract fetching."""
    
    def test_init(self):
        """Test fetcher initialization."""
        fetcher = PubMedFetcher(batch_size=50)
        assert fetcher.batch_size == 50
    
    @patch('gene_rif_graph.ingest.Entrez')
    def test_fetch_abstracts(self, mock_entrez):
        """Test abstract fetching with mocked Entrez."""
        # Mock Entrez response
        mock_handle = Mock()
        mock_entrez.efetch.return_value = mock_handle
        mock_entrez.read.return_value = {
            'PubmedArticle': [
                {
                    'MedlineCitation': {
                        'PMID': 12345,
                        'Article': {
                            'ArticleTitle': 'Test Article',
                            'Abstract': {
                                'AbstractText': ['Test abstract text.']
                            }
                        }
                    }
                }
            ]
        }
        
        fetcher = PubMedFetcher(batch_size=10)
        abstracts = fetcher.fetch_abstracts([12345])
        
        assert 12345 in abstracts
        assert abstracts[12345]['title'] == 'Test Article'
        assert abstracts[12345]['abstract'] == 'Test abstract text.'
    
    def test_augment_generifs(self, sample_generifs):
        """Test augmenting GeneRIFs with abstracts."""
        df = pd.DataFrame(sample_generifs)
        
        # Mock fetcher
        fetcher = PubMedFetcher()
        fetcher.fetch_abstracts = Mock(return_value={
            12345: {'title': 'TP53 Study', 'abstract': 'TP53 abstract'},
            12346: {'title': 'BRCA1 Study', 'abstract': 'BRCA1 abstract'},
            12347: {'title': 'EGFR Study', 'abstract': 'EGFR abstract'}
        })
        
        augmented_df = fetcher.augment_generifs_with_abstracts(df)
        
        assert 'title' in augmented_df.columns
        assert 'abstract' in augmented_df.columns
        assert augmented_df['title'].notna().all()
        assert augmented_df['abstract'].notna().all()
