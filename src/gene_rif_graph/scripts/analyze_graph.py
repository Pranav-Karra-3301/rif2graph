#!/usr/bin/env python3
"""
Analyze knowledge graphs and generate reports.
"""

import logging
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
@click.option('--graph-file', required=True, help='Input graph file (pickle or graphml)')
@click.option('--output-dir', default='./data/graphs', help='Output directory')
@click.option('--top-k', default=20, type=int, help='Number of top hubs to report')
@click.option('--community-method', default='louvain', help='Community detection method')
@click.option('--generate-report', is_flag=True, help='Generate comprehensive report')
@click.option('--export-csv', is_flag=True, help='Export results to CSV files')
@click.option('--analyze-projections', is_flag=True, help='Analyze graph projections')
@click.option('--query-node', help='Query specific node for detailed analysis')
@click.option('--query-community', type=int, help='Query specific community')
def main(graph_file, output_dir, top_k, community_method, generate_report,
         export_csv, analyze_projections, query_node, query_community):
    """Analyze knowledge graphs and generate reports."""
    
    logger.info("Starting graph analysis")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load graph
        logger.info(f"Loading graph from {graph_file}")
        graph_builder = GraphBuilder(output_dir=output_dir)
        
        graph_path = Path(graph_file)
        if graph_path.suffix == '.pkl':
            graph = graph_builder.load_graph(graph_path.stem, format='pickle')
        elif graph_path.suffix == '.graphml':
            graph = graph_builder.load_graph(graph_path.stem, format='graphml')
        else:
            raise click.ClickException(f"Unsupported graph format: {graph_path.suffix}")
        
        logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Initialize analyzer
        analyzer = GraphAnalyzer(output_dir=output_dir)
        
        # Basic statistics
        logger.info("Computing basic statistics...")
        stats = analyzer.compute_basic_stats(graph)
        
        print("\n" + "="*50)
        print("GRAPH STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Find top hubs
        logger.info("Finding top hub nodes...")
        top_hubs = analyzer.find_top_hubs(graph, top_k=top_k)
        
        print(f"\nTOP {len(top_hubs)} HUB NODES:")
        print("-" * 30)
        for i, (node, score) in enumerate(top_hubs, 1):
            node_type = graph.nodes.get(node, {}).get('node_type', 'unknown')
            print(f"{i:2d}. {node} ({node_type}): {score:.4f}")
        
        # Community detection
        logger.info(f"Detecting communities using {community_method}...")
        communities = analyzer.detect_communities(graph, method=community_method)
        
        print(f"\nCOMMUNITY DETECTION ({community_method.upper()}):")
        print("-" * 40)
        print(f"Number of communities: {len(communities)}")
        
        # Analyze communities
        comm_stats = analyzer.analyze_communities(graph, communities)
        
        if not comm_stats.empty:
            print(f"Average community size: {comm_stats['size'].mean():.2f}")
            print(f"Largest community: {comm_stats['size'].max()} nodes")
            print(f"Smallest community: {comm_stats['size'].min()} nodes")
            
            print("\nTop 10 communities by size:")
            top_communities = comm_stats.nlargest(10, 'size')
            for _, row in top_communities.iterrows():
                comm_id = int(row['community_id'])
                size = int(row['size'])
                edges = int(row['edges'])
                density = row['density']
                print(f"  Community {comm_id}: {size} nodes, {edges} edges, density={density:.3f}")
        
        # Query specific node if requested
        if query_node:
            print(f"\nNODE ANALYSIS: {query_node}")
            print("-" * 30)
            
            if query_node in graph:
                node_data = graph.nodes[query_node]
                degree = graph.degree(query_node)
                neighbors = list(graph.neighbors(query_node))
                
                print(f"Type: {node_data.get('node_type', 'unknown')}")
                print(f"Degree: {degree}")
                print(f"Bipartite set: {node_data.get('bipartite', 'unknown')}")
                
                if neighbors:
                    print(f"Top 10 neighbors:")
                    # Sort neighbors by edge weight if available
                    neighbor_weights = []
                    for neighbor in neighbors:
                        edge_data = graph[query_node][neighbor]
                        weight = edge_data.get('weight', 1)
                        neighbor_weights.append((neighbor, weight))
                    
                    neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                    for neighbor, weight in neighbor_weights[:10]:
                        neighbor_type = graph.nodes.get(neighbor, {}).get('node_type', 'unknown')
                        print(f"  {neighbor} ({neighbor_type}): weight={weight}")
            else:
                print(f"Node '{query_node}' not found in graph")
        
        # Query specific community if requested
        if query_community is not None:
            print(f"\nCOMMUNITY ANALYSIS: Community {query_community}")
            print("-" * 40)
            
            if query_community in communities:
                community_nodes = communities[query_community]
                subgraph = graph.subgraph(community_nodes)
                
                print(f"Size: {len(community_nodes)} nodes")
                print(f"Edges: {subgraph.number_of_edges()}")
                print(f"Density: {subgraph.number_of_edges() / (len(community_nodes) * (len(community_nodes) - 1) / 2) if len(community_nodes) > 1 else 0:.3f}")
                
                # Show node types in community
                node_types = {}
                for node in community_nodes:
                    node_type = graph.nodes.get(node, {}).get('node_type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                print("Node type distribution:")
                for node_type, count in node_types.items():
                    print(f"  {node_type}: {count}")
                
                # Show top nodes in community by degree
                community_degrees = [(node, subgraph.degree(node)) for node in community_nodes]
                community_degrees.sort(key=lambda x: x[1], reverse=True)
                
                print("Top 10 nodes by degree:")
                for node, degree in community_degrees[:10]:
                    node_type = graph.nodes.get(node, {}).get('node_type', 'unknown')
                    print(f"  {node} ({node_type}): degree={degree}")
            else:
                print(f"Community {query_community} not found")
        
        # Export CSV files if requested
        if export_csv:
            logger.info("Exporting results to CSV...")
            
            # Export top hubs
            hubs_df = pd.DataFrame(top_hubs, columns=['node', 'centrality_score'])
            hubs_df['node_type'] = hubs_df['node'].map(
                lambda x: graph.nodes.get(x, {}).get('node_type', 'unknown')
            )
            hubs_df['degree'] = hubs_df['node'].map(lambda x: graph.degree(x))
            
            hubs_csv = output_dir / "top_hubs.csv"
            hubs_df.to_csv(hubs_csv, index=False)
            print(f"Exported top hubs to: {hubs_csv}")
            
            # Export community assignments
            if communities:
                analyzer.export_community_csv(graph, communities, "community_assignments.csv")
                
                # Export community statistics
                if not comm_stats.empty:
                    comm_csv = output_dir / "community_statistics.csv"
                    comm_stats.to_csv(comm_csv, index=False)
                    print(f"Exported community stats to: {comm_csv}")
            
            # Export edge list
            edges_data = []
            for u, v, data in graph.edges(data=True):
                edge_row = {
                    'source': u,
                    'target': v,
                    'weight': data.get('weight', 1),
                    'source_type': graph.nodes.get(u, {}).get('node_type', 'unknown'),
                    'target_type': graph.nodes.get(v, {}).get('node_type', 'unknown')
                }
                
                # Add predicates if available
                predicates = data.get('predicates', [])
                if predicates:
                    edge_row['predicates'] = '; '.join(predicates)
                
                edges_data.append(edge_row)
            
            edges_df = pd.DataFrame(edges_data)
            edges_csv = output_dir / "edges.csv"
            edges_df.to_csv(edges_csv, index=False)
            print(f"Exported edges to: {edges_csv}")
        
        # Analyze projections if requested
        if analyze_projections:
            logger.info("Analyzing graph projections...")
            
            # Check if graph is bipartite
            if stats.get('is_bipartite', False):
                try:
                    # Create and analyze gene projection
                    gene_graph = graph_builder.project_gene_graph(graph)
                    gene_stats = analyzer.compute_basic_stats(gene_graph)
                    
                    print(f"\nGENE PROJECTION ANALYSIS:")
                    print("-" * 30)
                    for key, value in gene_stats.items():
                        print(f"{key.replace('_', ' ').title()}: {value}")
                    
                    # Find top gene hubs
                    gene_hubs = analyzer.find_top_hubs(gene_graph, top_k=10)
                    print("Top 10 gene hubs:")
                    for i, (node, score) in enumerate(gene_hubs, 1):
                        print(f"  {i:2d}. {node}: {score:.4f}")
                    
                    # Create and analyze function projection
                    function_graph = graph_builder.project_function_graph(graph)
                    function_stats = analyzer.compute_basic_stats(function_graph)
                    
                    print(f"\nFUNCTION PROJECTION ANALYSIS:")
                    print("-" * 30)
                    for key, value in function_stats.items():
                        print(f"{key.replace('_', ' ').title()}: {value}")
                    
                    # Find top function hubs
                    function_hubs = analyzer.find_top_hubs(function_graph, top_k=10)
                    print("Top 10 function hubs:")
                    for i, (node, score) in enumerate(function_hubs, 1):
                        print(f"  {i:2d}. {node}: {score:.4f}")
                
                except Exception as e:
                    logger.warning(f"Error analyzing projections: {e}")
            else:
                print("\nGraph is not bipartite - skipping projection analysis")
        
        # Generate comprehensive report if requested
        if generate_report:
            logger.info("Generating comprehensive report...")
            report_path = analyzer.generate_summary_report(graph)
            print(f"\nComprehensive report saved to: {report_path}")
        
        print("\nGraph analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during graph analysis: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
