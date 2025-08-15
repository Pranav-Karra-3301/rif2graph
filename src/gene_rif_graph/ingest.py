"""
Data ingestion module for downloading and parsing GeneRIF data from NCBI.
"""

import os
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from urllib.parse import urljoin
import requests
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GeneRIFDownloader:
    """Download and parse GeneRIF data from NCBI FTP."""
    
    GENERIF_URL = "https://ftp.ncbi.nlm.nih.gov/gene/GeneRIF/"
    GENERIF_FILES = {
        "human": "generifs_basic.gz",
        "all": "generifs_basic.gz"
    }
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup NCBI email for Entrez
        self.email = os.getenv("NCBI_EMAIL")
        if self.email:
            Entrez.email = self.email
        
        self.api_key = os.getenv("NCBI_API_KEY")
        if self.api_key:
            Entrez.api_key = self.api_key
    
    def download_generifs(self, species: str = "all", force_download: bool = False) -> Path:
        """
        Download GeneRIF data from NCBI FTP.
        
        Args:
            species: Species to download ('human' or 'all')
            force_download: Whether to re-download existing files
            
        Returns:
            Path to downloaded file
        """
        if species not in self.GENERIF_FILES:
            raise ValueError(f"Species {species} not supported. Use 'human' or 'all'")
        
        filename = self.GENERIF_FILES[species]
        url = urljoin(self.GENERIF_URL, filename)
        local_path = self.data_dir / filename
        
        if local_path.exists() and not force_download:
            logger.info(f"GeneRIF file already exists: {local_path}")
            return local_path
        
        logger.info(f"Downloading GeneRIF data from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded GeneRIF data to {local_path}")
        return local_path
    
    def parse_generifs(self, file_path: Path, species_filter: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Parse GeneRIF file into pandas DataFrame.
        
        Args:
            file_path: Path to GeneRIF file
            species_filter: List of taxonomy IDs to filter by (e.g., [9606] for human)
            
        Returns:
            DataFrame with columns: tax_id, gene_id, pmid, timestamp, text
        """
        logger.info(f"Parsing GeneRIF file: {file_path}")
        
        # Column names for GeneRIF basic file
        columns = ['tax_id', 'gene_id', 'pmid', 'timestamp', 'text']
        
        # Read compressed file
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                # Skip header line
                next(f)
                data = []
                for line in tqdm(f, desc="Parsing GeneRIFs"):
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        data.append(parts[:5])
        else:
            data = pd.read_csv(file_path, sep='\t', names=columns, skiprows=1)
        
        df = pd.DataFrame(data, columns=columns)
        
        # Convert data types
        df['tax_id'] = pd.to_numeric(df['tax_id'], errors='coerce')
        df['gene_id'] = pd.to_numeric(df['gene_id'], errors='coerce')
        df['pmid'] = pd.to_numeric(df['pmid'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Filter by species if specified
        if species_filter:
            df = df[df['tax_id'].isin(species_filter)]
            logger.info(f"Filtered to {len(df)} GeneRIFs for species: {species_filter}")
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['gene_id', 'text'])
        
        logger.info(f"Parsed {len(df)} GeneRIFs")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "generifs_processed.csv") -> Path:
        """Save processed GeneRIF data to CSV."""
        processed_dir = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        return output_path


class PubMedFetcher:
    """Fetch PubMed abstracts for GeneRIF PMIDs."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        
        # Setup NCBI credentials
        self.email = os.getenv("NCBI_EMAIL")
        if self.email:
            Entrez.email = self.email
        
        self.api_key = os.getenv("NCBI_API_KEY")
        if self.api_key:
            Entrez.api_key = self.api_key
    
    def fetch_abstracts(self, pmids: List[int]) -> Dict[int, Dict]:
        """
        Fetch PubMed abstracts for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            Dictionary mapping PMID to abstract data
        """
        abstracts = {}
        
        # Process in batches to avoid API limits
        for i in tqdm(range(0, len(pmids), self.batch_size), desc="Fetching abstracts"):
            batch_pmids = pmids[i:i + self.batch_size]
            batch_abstracts = self._fetch_batch(batch_pmids)
            abstracts.update(batch_abstracts)
        
        return abstracts
    
    def _fetch_batch(self, pmids: List[int]) -> Dict[int, Dict]:
        """Fetch a batch of abstracts."""
        try:
            pmid_str = ','.join(map(str, pmids))
            
            # Fetch records from PubMed
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid_str,
                rettype="abstract",
                retmode="xml"
            )
            
            records = Entrez.read(handle)
            handle.close()
            
            abstracts = {}
            
            for record in records['PubmedArticle']:
                try:
                    pmid = int(record['MedlineCitation']['PMID'])
                    article = record['MedlineCitation']['Article']
                    
                    # Extract title
                    title = article.get('ArticleTitle', '')
                    
                    # Extract abstract
                    abstract_text = ''
                    if 'Abstract' in article:
                        abstract_parts = article['Abstract'].get('AbstractText', [])
                        if isinstance(abstract_parts, list):
                            abstract_text = ' '.join(str(part) for part in abstract_parts)
                        else:
                            abstract_text = str(abstract_parts)
                    
                    abstracts[pmid] = {
                        'title': title,
                        'abstract': abstract_text,
                        'pmid': pmid
                    }
                    
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")
                    continue
            
            return abstracts
            
        except Exception as e:
            logger.error(f"Error fetching batch {pmids[:5]}...: {e}")
            return {}
    
    def augment_generifs_with_abstracts(self, generifs_df: pd.DataFrame) -> pd.DataFrame:
        """Add PubMed abstracts to GeneRIF DataFrame."""
        unique_pmids = generifs_df['pmid'].dropna().unique().astype(int).tolist()
        logger.info(f"Fetching abstracts for {len(unique_pmids)} unique PMIDs")
        
        abstracts = self.fetch_abstracts(unique_pmids)
        
        # Add abstract data to DataFrame
        generifs_df['title'] = generifs_df['pmid'].map(
            lambda x: abstracts.get(int(x), {}).get('title', '') if pd.notna(x) else ''
        )
        generifs_df['abstract'] = generifs_df['pmid'].map(
            lambda x: abstracts.get(int(x), {}).get('abstract', '') if pd.notna(x) else ''
        )
        
        logger.info(f"Added abstracts for {sum(generifs_df['abstract'] != '')} GeneRIFs")
        return generifs_df
