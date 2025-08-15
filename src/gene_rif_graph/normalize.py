"""
Normalization utilities for genes and biomedical concepts.
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from Bio import Entrez
import pickle
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GeneNormalizer:
    """Normalize gene IDs and symbols using NCBI data."""
    
    def __init__(self, cache_dir: str = "./data/processed"):
        """
        Initialize gene normalizer.
        
        Args:
            cache_dir: Directory to store cached gene mappings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Gene mapping caches
        self.entrez_to_symbol = {}
        self.symbol_to_entrez = {}
        self.alias_to_official = {}
        self.species_genes = {}
        
        # Setup NCBI credentials
        self.email = os.getenv("NCBI_EMAIL")
        if self.email:
            Entrez.email = self.email
        
        self.api_key = os.getenv("NCBI_API_KEY")
        if self.api_key:
            Entrez.api_key = self.api_key
        
        self._load_cached_mappings()
    
    def _load_cached_mappings(self):
        """Load cached gene mappings if available."""
        cache_file = self.cache_dir / "gene_mappings.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.entrez_to_symbol = cache_data.get('entrez_to_symbol', {})
                    self.symbol_to_entrez = cache_data.get('symbol_to_entrez', {})
                    self.alias_to_official = cache_data.get('alias_to_official', {})
                    self.species_genes = cache_data.get('species_genes', {})
                
                logger.info(f"Loaded cached gene mappings: {len(self.entrez_to_symbol)} genes")
            except Exception as e:
                logger.warning(f"Error loading cached mappings: {e}")
    
    def _save_cached_mappings(self):
        """Save gene mappings to cache."""
        cache_file = self.cache_dir / "gene_mappings.pkl"
        
        cache_data = {
            'entrez_to_symbol': self.entrez_to_symbol,
            'symbol_to_entrez': self.symbol_to_entrez,
            'alias_to_official': self.alias_to_official,
            'species_genes': self.species_genes
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved gene mappings cache: {len(self.entrez_to_symbol)} genes")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def download_gene_info(self, species_tax_ids: List[int] = [9606]) -> Path:
        """
        Download gene info from NCBI for specified species.
        
        Args:
            species_tax_ids: List of taxonomy IDs (default: [9606] for human)
            
        Returns:
            Path to downloaded gene info file
        """
        gene_info_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
        local_path = self.cache_dir / "gene_info.gz"
        
        if not local_path.exists():
            logger.info("Downloading NCBI gene_info.gz...")
            response = requests.get(gene_info_url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Parse gene info for specified species
        self._parse_gene_info(local_path, species_tax_ids)
        return local_path
    
    def _parse_gene_info(self, gene_info_path: Path, species_tax_ids: List[int]):
        """Parse gene_info file and build mappings."""
        import gzip
        
        logger.info("Parsing gene_info file...")
        
        with gzip.open(gene_info_path, 'rt', encoding='utf-8') as f:
            header = next(f).strip().split('\t')
            
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 15:
                    continue
                
                tax_id = int(fields[0])
                if tax_id not in species_tax_ids:
                    continue
                
                gene_id = int(fields[1])
                symbol = fields[2]
                synonyms = fields[4].split('|') if fields[4] != '-' else []
                
                # Build mappings
                self.entrez_to_symbol[gene_id] = symbol
                self.symbol_to_entrez[symbol] = gene_id
                
                # Add species mapping
                if tax_id not in self.species_genes:
                    self.species_genes[tax_id] = set()
                self.species_genes[tax_id].add(gene_id)
                
                # Add synonym mappings
                for synonym in synonyms:
                    if synonym and synonym != '-':
                        self.alias_to_official[synonym.upper()] = symbol.upper()
                        self.symbol_to_entrez[synonym] = gene_id
        
        logger.info(f"Parsed {len(self.entrez_to_symbol)} genes for species {species_tax_ids}")
        self._save_cached_mappings()
    
    def normalize_gene_id(self, gene_id: int) -> Optional[str]:
        """
        Convert Entrez Gene ID to official symbol.
        
        Args:
            gene_id: Entrez Gene ID
            
        Returns:
            Official gene symbol or None if not found
        """
        return self.entrez_to_symbol.get(gene_id)
    
    def normalize_gene_symbol(self, symbol: str) -> Optional[str]:
        """
        Normalize gene symbol to official symbol.
        
        Args:
            symbol: Gene symbol (potentially an alias)
            
        Returns:
            Official gene symbol or None if not found
        """
        symbol_upper = symbol.upper()
        
        # Check if it's already an official symbol
        if symbol in self.symbol_to_entrez:
            return symbol
        
        # Check aliases
        return self.alias_to_official.get(symbol_upper)
    
    def get_entrez_id(self, symbol: str) -> Optional[int]:
        """
        Get Entrez Gene ID for a gene symbol.
        
        Args:
            symbol: Gene symbol
            
        Returns:
            Entrez Gene ID or None if not found
        """
        return self.symbol_to_entrez.get(symbol)
    
    def batch_normalize_genes(self, gene_identifiers: List[str]) -> Dict[str, Optional[str]]:
        """
        Normalize multiple gene identifiers.
        
        Args:
            gene_identifiers: List of gene symbols or IDs
            
        Returns:
            Dictionary mapping input to normalized symbol
        """
        results = {}
        
        for identifier in gene_identifiers:
            # Try as symbol first
            normalized = self.normalize_gene_symbol(identifier)
            
            # Try as Entrez ID if symbol normalization failed
            if not normalized:
                try:
                    gene_id = int(identifier)
                    normalized = self.normalize_gene_id(gene_id)
                except ValueError:
                    pass
            
            results[identifier] = normalized
        
        return results


class ConceptNormalizer:
    """Normalize biomedical concepts using UMLS and MeSH."""
    
    def __init__(self, cache_dir: str = "./data/processed"):
        """
        Initialize concept normalizer.
        
        Args:
            cache_dir: Directory to store cached concept mappings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Concept mappings
        self.umls_to_preferred = {}
        self.mesh_to_preferred = {}
        self.concept_types = {}
        
        self._load_cached_concepts()
    
    def _load_cached_concepts(self):
        """Load cached concept mappings."""
        cache_file = self.cache_dir / "concept_mappings.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.umls_to_preferred = cache_data.get('umls_to_preferred', {})
                    self.mesh_to_preferred = cache_data.get('mesh_to_preferred', {})
                    self.concept_types = cache_data.get('concept_types', {})
                
                logger.info(f"Loaded cached concept mappings: {len(self.umls_to_preferred)} concepts")
            except Exception as e:
                logger.warning(f"Error loading cached concept mappings: {e}")
    
    def _save_cached_concepts(self):
        """Save concept mappings to cache."""
        cache_file = self.cache_dir / "concept_mappings.pkl"
        
        cache_data = {
            'umls_to_preferred': self.umls_to_preferred,
            'mesh_to_preferred': self.mesh_to_preferred,
            'concept_types': self.concept_types
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved concept mappings cache")
        except Exception as e:
            logger.error(f"Error saving concept cache: {e}")
    
    def normalize_umls_concept(self, umls_id: str) -> Optional[Dict[str, str]]:
        """
        Normalize UMLS concept to preferred term.
        
        Args:
            umls_id: UMLS Concept ID (CUI)
            
        Returns:
            Dictionary with preferred term and semantic type
        """
        if umls_id in self.umls_to_preferred:
            return self.umls_to_preferred[umls_id]
        
        # In a production system, this would query UMLS API
        # For now, return a placeholder
        return {
            'preferred_term': f"UMLS:{umls_id}",
            'semantic_type': 'Unknown'
        }
    
    def normalize_text_concept(self, text: str, entity_type: str = None) -> Optional[str]:
        """
        Normalize text concept to standard form.
        
        Args:
            text: Concept text
            entity_type: Entity type hint
            
        Returns:
            Normalized concept text
        """
        # Basic text normalization
        normalized = text.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['the ', 'a ', 'an ']
        suffixes_to_remove = [' protein', ' gene', ' receptor']
        
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Disease-specific normalization
        if entity_type in ['DISEASE', 'DISORDER']:
            # Remove "syndrome" from disease names for consistency
            normalized = normalized.replace(' syndrome', '')
            normalized = normalized.replace(' disease', '')
        
        # Chemical-specific normalization
        elif entity_type in ['CHEMICAL', 'DRUG']:
            # Remove dosage information
            import re
            normalized = re.sub(r'\d+\s*(mg|mcg|g|ml|l)\b', '', normalized)
        
        return normalized.title()  # Title case for consistency
    
    def create_concept_clusters(self, concepts: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Group similar concepts together.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            Dictionary mapping cluster representative to list of concepts
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not concepts:
            return {}
        
        # Extract concept texts
        texts = [c.get('text', '') for c in concepts]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        try:
            vectors = vectorizer.fit_transform(texts)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(vectors)
            
            # Cluster concepts
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.3,
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(1 - similarity_matrix)
            
            # Group concepts by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(concepts[i])
            
            # Select representative for each cluster (most frequent or shortest)
            cluster_representatives = {}
            for label, cluster_concepts in clusters.items():
                # Use shortest text as representative
                representative = min(cluster_concepts, key=lambda x: len(x.get('text', '')))
                cluster_representatives[representative['text']] = cluster_concepts
            
            return cluster_representatives
            
        except Exception as e:
            logger.warning(f"Error clustering concepts: {e}")
            # Fallback: each concept is its own cluster
            return {c.get('text', f'concept_{i}'): [c] for i, c in enumerate(concepts)}
    
    def batch_normalize_concepts(self, concepts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize multiple concepts.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            List of normalized concept dictionaries
        """
        normalized_concepts = []
        
        for concept in concepts:
            normalized = concept.copy()
            
            # Normalize text
            if 'text' in concept:
                normalized['normalized_text'] = self.normalize_text_concept(
                    concept['text'], 
                    concept.get('label')
                )
            
            # Normalize UMLS ID if present
            if 'umls_id' in concept and concept['umls_id']:
                umls_info = self.normalize_umls_concept(concept['umls_id'])
                if umls_info:
                    normalized.update(umls_info)
            
            normalized_concepts.append(normalized)
        
        return normalized_concepts
