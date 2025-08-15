#!/usr/bin/env python3
"""
Download and process GeneRIF data from NCBI.
"""

import os
import logging
import click
from pathlib import Path
from gene_rif_graph.ingest import GeneRIFDownloader, PubMedFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--species', default='all', help='Species to download (human/all)')
@click.option('--data-dir', default='./data/raw', help='Directory to store raw data')
@click.option('--force', is_flag=True, help='Force re-download existing files')
@click.option('--fetch-abstracts', is_flag=True, help='Fetch PubMed abstracts')
@click.option('--species-filter', multiple=True, type=int, 
              help='Taxonomy IDs to filter (e.g., 9606 for human)')
@click.option('--output-file', default='generifs_processed.csv', 
              help='Output filename for processed data')
def main(species, data_dir, force, fetch_abstracts, species_filter, output_file):
    """Download and process GeneRIF data from NCBI."""
    
    logger.info("Starting GeneRIF data download and processing")
    
    # Initialize downloader
    downloader = GeneRIFDownloader(data_dir=data_dir)
    
    try:
        # Download GeneRIF data
        generif_file = downloader.download_generifs(species=species, force_download=force)
        
        # Parse GeneRIF data
        species_filter_list = list(species_filter) if species_filter else None
        generifs_df = downloader.parse_generifs(generif_file, species_filter=species_filter_list)
        
        logger.info(f"Parsed {len(generifs_df)} GeneRIFs")
        
        # Fetch PubMed abstracts if requested
        if fetch_abstracts:
            logger.info("Fetching PubMed abstracts...")
            fetcher = PubMedFetcher()
            generifs_df = fetcher.augment_generifs_with_abstracts(generifs_df)
        
        # Save processed data
        output_path = downloader.save_processed_data(generifs_df, filename=output_file)
        
        # Print summary statistics
        logger.info("Download and processing completed successfully!")
        logger.info(f"Total GeneRIFs: {len(generifs_df)}")
        logger.info(f"Unique genes: {generifs_df['gene_id'].nunique()}")
        logger.info(f"Unique PMIDs: {generifs_df['pmid'].nunique()}")
        if 'tax_id' in generifs_df.columns:
            logger.info(f"Species distribution:")
            for tax_id, count in generifs_df['tax_id'].value_counts().head().items():
                logger.info(f"  Tax ID {tax_id}: {count} GeneRIFs")
        
        logger.info(f"Processed data saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during download/processing: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
