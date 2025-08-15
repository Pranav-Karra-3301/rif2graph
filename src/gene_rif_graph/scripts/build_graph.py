#!/usr/bin/env python3
"""
Build knowledge graphs from extracted relation triplets.
"""

import os
import logging
import pickle
import click
import pandas as pd
from pathlib import Path
from gene_rif_graph.graph import GraphBuilder, GraphAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--relations-file', required=True, help='Input relations file (pickle or CSV)')
@click.option('--output-dir', default='./data/graphs', help='Output directory for graphs')
@click.option('--min-edge-weight', default=1, type=int, help='Minimum edge weight threshold')
@click.option('--max-nodes', default=50000, type=int, help='Maximum number of nodes')
@click.option('--save-format', default='both', help='Save format (graphml/pickle/both)')
@click.option('--create-projections', is_flag=True, help='Create gene and function projections')
@click.option('--analyze', is_flag=True, help='Run graph analysis')
@click.option('--summary-report', is_flag=True, help='Generate summary report')
def main(relations_file, output_dir, min_edge_weight, max_nodes, save_format,
         create_projections, analyze, summary_report):
    """Build knowledge graphs from extracted relation triplets."""
    
    logger.info("Starting graph construction")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for graph builder
    os.environ["MIN_EDGE_WEIGHT"] = str(min_edge_weight)
    os.environ["MAX_NODES"] = str(max_nodes)
    
    try:
        # Load relations data
        logger.info(f"Loading relations from {relations_file}")
        relations_path = Path(relations_file)
        
        if relations_path.suffix == '.pkl':
            with open(relations_path, 'rb') as f:
                relations = pickle.load(f)
        elif relations_path.suffix == '.csv':
            # Load from CSV and reconstruct relation objects
            df = pd.read_csv(relations_path)
            relations = []
            
            for _, row in df.iterrows():
                rel = {
                    'subject': {
                        'text': row['subject_text'],
                        'label': row['subject_label']
                    },
                    'predicate': row['predicate'],
                    'object': {
                        'text': row['object_text'],
                        'label': row['object_label']
                    },
                    'confidence': row.get('confidence', 1.0)
                }
                
                # Add normalized texts if available
                if 'subject_normalized' in row and pd.notna(row['subject_normalized']):
                    rel['subject']['normalized_text'] = row['subject_normalized']
                if 'object_normalized' in row and pd.notna(row['object_normalized']):
                    rel['object']['normalized_text'] = row['object_normalized']
                
                # Add UMLS IDs if available
                if 'subject_umls_id' in row and pd.notna(row['subject_umls_id']):
                    rel['subject']['umls_id'] = row['subject_umls_id']
                if 'object_umls_id' in row and pd.notna(row['object_umls_id']):
                    rel['object']['umls_id'] = row['object_umls_id']
                
                relations.append(rel)
        else:
            raise click.ClickException(f"Unsupported file format: {relations_path.suffix}")
        
        logger.info(f"Loaded {len(relations)} relations")
        
        # Initialize graph builder
        graph_builder = GraphBuilder(output_dir=output_dir)
        
        # Build bipartite graph
        logger.info("Building bipartite gene-function graph...")
        bipartite_graph = graph_builder.build_graph_from_triplets(relations)
        
        # Save bipartite graph
        graph_builder.save_graph(bipartite_graph, "bipartite_graph", format=save_format)
        
        # Create projections if requested
        if create_projections:
            logger.info("Creating graph projections...")
            
            # Gene projection
            gene_graph = graph_builder.project_gene_graph(bipartite_graph)
            graph_builder.save_graph(gene_graph, "gene_projection", format=save_format)
            
            # Function projection
            function_graph = graph_builder.project_function_graph(bipartite_graph)
            graph_builder.save_graph(function_graph, "function_projection", format=save_format)
        
        # Run analysis if requested
        if analyze or summary_report:
            logger.info("Analyzing graphs...")
            analyzer = GraphAnalyzer(output_dir=output_dir)
            
            # Analyze bipartite graph
            stats = analyzer.compute_basic_stats(bipartite_graph)
            logger.info("Bipartite graph statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            # Find top hubs
            top_hubs = analyzer.find_top_hubs(bipartite_graph, top_k=20)
            logger.info("Top 10 hub nodes:")
            for i, (node, score) in enumerate(top_hubs[:10], 1):
                logger.info(f"  {i:2d}. {node}: {score:.4f}")
            
            # Detect communities
            communities = analyzer.detect_communities(bipartite_graph)
            logger.info(f"Detected {len(communities)} communities")
            
            # Analyze communities
            comm_stats = analyzer.analyze_communities(bipartite_graph, communities)
            if not comm_stats.empty:
                logger.info("Top 5 communities by size:")
                top_communities = comm_stats.nlargest(5, 'size')
                for _, row in top_communities.iterrows():
                    logger.info(f"  Community {int(row['community_id'])}: {int(row['size'])} nodes")
            
            # Export community assignments
            analyzer.export_community_csv(bipartite_graph, communities)
            
            # Generate summary report if requested
            if summary_report:
                analyzer.generate_summary_report(bipartite_graph)
            
            # Analyze projections if they exist
            if create_projections:
                logger.info("Analyzing gene projection...")
                gene_stats = analyzer.compute_basic_stats(gene_graph)
                logger.info("Gene projection statistics:")
                for key, value in gene_stats.items():
                    logger.info(f"  {key}: {value}")
                
                logger.info("Analyzing function projection...")
                function_stats = analyzer.compute_basic_stats(function_graph)
                logger.info("Function projection statistics:")
                for key, value in function_stats.items():
                    logger.info(f"  {key}: {value}")
        
        # Print final summary
        logger.info("Graph construction completed successfully!")
        logger.info(f"Bipartite graph: {bipartite_graph.number_of_nodes()} nodes, {bipartite_graph.number_of_edges()} edges")
        
        if create_projections:
            logger.info(f"Gene projection: {gene_graph.number_of_nodes()} nodes, {gene_graph.number_of_edges()} edges")
            logger.info(f"Function projection: {function_graph.number_of_nodes()} nodes, {function_graph.number_of_edges()} edges")
        
        logger.info(f"Graphs saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during graph construction: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
