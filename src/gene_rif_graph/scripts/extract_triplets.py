#!/usr/bin/env python3
"""
Extract biomedical entities and relations from GeneRIF texts using NLP.
"""

import os
import logging
import pickle
import click
import pandas as pd
from pathlib import Path
from gene_rif_graph.nlp import BioNERExtractor, RelationExtractor
from gene_rif_graph.filters import TripleFilter, QualityFilter
from gene_rif_graph.normalize import GeneNormalizer, ConceptNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--input-file', required=True, help='Input CSV file with GeneRIF data')
@click.option('--output-dir', default='./data/processed', help='Output directory')
@click.option('--ner-model', default='en_ner_bionlp13cg_md', help='SciSpaCy NER model')
@click.option('--re-model', default='Babelscape/rebel-large', help='Relation extraction model')
@click.option('--device', default='auto', help='Device to use (auto/cpu/cuda)')
@click.option('--batch-size', default=100, type=int, help='Batch size for processing')
@click.option('--max-texts', type=int, help='Maximum number of texts to process (for testing)')
@click.option('--min-confidence', default=0.2, type=float, help='Minimum relation confidence')
@click.option('--normalize-genes', is_flag=True, help='Normalize gene identifiers')
@click.option('--normalize-concepts', is_flag=True, help='Normalize biomedical concepts')
@click.option('--filter-gene-gene', is_flag=True, default=True, help='Filter gene-gene relations')
@click.option('--text-column', default='text', help='Column name containing text to process')
def main(input_file, output_dir, ner_model, re_model, device, batch_size, max_texts,
         min_confidence, normalize_genes, normalize_concepts, filter_gene_gene, text_column):
    """Extract biomedical entities and relations from GeneRIF texts."""
    
    logger.info("Starting NLP pipeline for relation extraction")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load input data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        if text_column not in df.columns:
            raise click.ClickException(f"Column '{text_column}' not found in input file")
        
        # Limit data for testing
        if max_texts:
            df = df.head(max_texts)
            logger.info(f"Limited to {max_texts} texts for processing")
        
        texts = df[text_column].dropna().tolist()
        logger.info(f"Processing {len(texts)} texts")
        
        # Initialize NLP models
        logger.info("Loading NLP models...")
        ner_extractor = BioNERExtractor(model_name=ner_model)
        re_extractor = RelationExtractor(model_name=re_model, device=device)
        
        # Extract entities
        logger.info("Extracting biomedical entities...")
        entities_list = ner_extractor.batch_extract_entities(texts)
        
        # Checkpoint: Entity extraction stats
        total_entities = sum(len(ents) for ents in entities_list)
        logger.info(f"CHECKPOINT: Extracted {total_entities} entities from {len(texts)} texts ({total_entities/len(texts):.1f} entities/text)")
        
        # Save entities
        entities_file = output_dir / "entities.pkl"
        with open(entities_file, 'wb') as f:
            pickle.dump(entities_list, f)
        logger.info(f"Saved entities to {entities_file}")
        
        # Extract relations
        logger.info("Extracting relations...")
        relations_list = re_extractor.batch_extract_relations(texts, entities_list)
        
        # Checkpoint: Raw relation extraction stats
        total_raw_relations = sum(len(rels) for rels in relations_list)
        logger.info(f"CHECKPOINT: Extracted {total_raw_relations} raw relations from {len(texts)} texts ({total_raw_relations/len(texts):.1f} relations/text)")
        
        # Flatten relations list
        all_relations = []
        for i, relations in enumerate(relations_list):
            for rel in relations:
                rel['source_index'] = i
                rel['source_text'] = texts[i]
                if 'gene_id' in df.columns and i < len(df):
                    rel['gene_id'] = df.iloc[i]['gene_id']
                all_relations.append(rel)
        
        logger.info(f"CHECKPOINT: Flattened to {len(all_relations)} total relations")
        
        # Apply filters
        logger.info("Applying filters...")
        triple_filter = TripleFilter(
            min_confidence=min_confidence,
            filter_gene_gene=filter_gene_gene
        )
        quality_filter = QualityFilter()
        
        # Filter relations with checkpoints
        logger.info(f"CHECKPOINT: Before confidence filter: {len(all_relations)} relations")
        filtered_relations = triple_filter.filter_triplets(all_relations)
        logger.info(f"CHECKPOINT: After TripleFilter: {len(filtered_relations)} relations ({len(filtered_relations)/len(all_relations)*100:.1f}% retention)")
        
        filtered_relations = quality_filter.apply_quality_filters(filtered_relations)
        logger.info(f"CHECKPOINT: After QualityFilter: {len(filtered_relations)} relations")
        
        # Normalize predicates
        filtered_relations = triple_filter.normalize_predicates(filtered_relations)
        
        logger.info(f"CHECKPOINT: Final relations after all processing: {len(filtered_relations)} relations")
        
        # Check for concerning drop rates
        if len(all_relations) > 0:
            retention_rate = len(filtered_relations) / len(all_relations)
            if retention_rate < 0.1:
                logger.warning(f"WARNING: Very low retention rate ({retention_rate*100:.1f}%) - filters may be too aggressive!")
        
        # Initialize normalizers if requested
        gene_normalizer = None
        concept_normalizer = None
        
        if normalize_genes:
            logger.info("Initializing gene normalizer...")
            gene_normalizer = GeneNormalizer()
            gene_normalizer.download_gene_info()
        
        if normalize_concepts:
            logger.info("Initializing concept normalizer...")
            concept_normalizer = ConceptNormalizer()
        
        # Normalize entities in relations
        if gene_normalizer or concept_normalizer:
            logger.info("Normalizing entities...")
            normalized_relations = []
            
            for rel in filtered_relations:
                rel_copy = rel.copy()
                
                # Normalize subject
                if gene_normalizer and rel['subject']['label'] in ['GENE', 'PROTEIN']:
                    normalized_symbol = gene_normalizer.normalize_gene_symbol(rel['subject']['text'])
                    if normalized_symbol:
                        rel_copy['subject']['normalized_text'] = normalized_symbol
                
                # Normalize object
                if concept_normalizer:
                    normalized_concepts = concept_normalizer.batch_normalize_concepts([rel['object']])
                    if normalized_concepts:
                        rel_copy['object'].update(normalized_concepts[0])
                
                normalized_relations.append(rel_copy)
            
            filtered_relations = normalized_relations
        
        # Save relations
        relations_file = output_dir / "relations.pkl"
        with open(relations_file, 'wb') as f:
            pickle.dump(filtered_relations, f)
        logger.info(f"Saved relations to {relations_file}")
        
        # Create relations DataFrame for CSV export
        relation_rows = []
        for rel in filtered_relations:
            row = {
                'subject_text': rel['subject']['text'],
                'subject_label': rel['subject']['label'],
                'predicate': rel['predicate'],
                'object_text': rel['object']['text'],
                'object_label': rel['object']['label'],
                'confidence': rel.get('confidence', 1.0),
                'source_index': rel.get('source_index', -1)
            }
            
            # Add normalized texts if available
            if 'normalized_text' in rel['subject']:
                row['subject_normalized'] = rel['subject']['normalized_text']
            if 'normalized_text' in rel['object']:
                row['object_normalized'] = rel['object']['normalized_text']
            
            # Add UMLS IDs if available
            if 'umls_id' in rel['subject']:
                row['subject_umls_id'] = rel['subject']['umls_id']
            if 'umls_id' in rel['object']:
                row['object_umls_id'] = rel['object']['umls_id']
            
            # Add gene ID if available
            if 'gene_id' in rel:
                row['gene_id'] = rel['gene_id']
            
            relation_rows.append(row)
        
        # Save CSV
        relations_df = pd.DataFrame(relation_rows)
        relations_csv = output_dir / "relations.csv"
        relations_df.to_csv(relations_csv, index=False)
        logger.info(f"Saved relations CSV to {relations_csv}")
        
        # Generate summary statistics
        logger.info("Summary statistics:")
        logger.info(f"Total entities extracted: {sum(len(ents) for ents in entities_list)}")
        logger.info(f"Total relations extracted: {len(all_relations)}")
        logger.info(f"Relations after filtering: {len(filtered_relations)}")
        
        # Entity type distribution
        entity_types = {}
        for entities in entities_list:
            for ent in entities:
                label = ent['label']
                entity_types[label] = entity_types.get(label, 0) + 1
        
        logger.info("Entity type distribution:")
        for label, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {label}: {count}")
        
        # Predicate distribution
        predicate_counts = {}
        for rel in filtered_relations:
            pred = rel['predicate']
            predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
        
        logger.info("Top predicates:")
        for pred, count in sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {pred}: {count}")
        
        logger.info("NLP pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during NLP processing: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
