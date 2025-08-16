"""
Graph construction and analysis utilities.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional, Union
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build bipartite gene-function knowledge graphs from relation triplets."""
    
    def __init__(self, output_dir: str = "./data/graphs"):
        """
        Initialize graph builder.
        
        Args:
            output_dir: Directory to save graphs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph = None
        self.min_edge_weight = int(os.getenv("MIN_EDGE_WEIGHT", 1))
        self.max_nodes = int(os.getenv("MAX_NODES", 50000))
    
    def build_graph_from_triplets(self, relations: List[Dict[str, Any]]) -> nx.Graph:
        """
        Build a bipartite graph from relation triplets.
        
        Args:
            relations: List of relation dictionaries
            
        Returns:
            NetworkX bipartite graph
        """
        logger.info(f"Building graph from {len(relations)} relations")
        
        # Create bipartite graph
        G = nx.Graph()
        
        # Track edge weights (co-occurrence counts)
        edge_weights = defaultdict(int)
        
        # Add nodes and edges
        gene_nodes = set()
        function_nodes = set()
        
        for rel in relations:
            # Handle both nested dict format and flat CSV format
            if 'subject' in rel and isinstance(rel['subject'], dict):
                # Nested format
                subject = rel['subject']
                obj = rel['object']
                predicate = rel['predicate']
                confidence = rel.get('confidence', 1.0)
                
                subj_text = subject.get('normalized_text', subject['text'])
                obj_text = obj.get('normalized_text', obj['text'])
                subj_label = subject['label']
                obj_label = obj['label']
            else:
                # Flat CSV format
                subj_text = rel['subject_text']
                obj_text = rel['object_text']
                predicate = rel['predicate']
                confidence = rel.get('confidence', 1.0)
                subj_label = rel['subject_label']
                obj_label = rel['object_label']
            
            # Gene nodes (subjects should typically be genes, but may be classified as chemicals)
            if subj_label in ['GENE', 'PROTEIN', 'CHEMICAL']:
                gene_nodes.add(subj_text)
            
            # Function nodes (objects should be functions/diseases/chemicals)
            if obj_label in ['DISEASE', 'DISORDER', 'CHEMICAL', 'DRUG', 'FUNCTION', 'PROCESS']:
                function_nodes.add(obj_text)
            
            # Add edge with weight based on confidence and frequency
            edge_key = (subj_text, obj_text)
            edge_weights[edge_key] += confidence
            
            # Store relation metadata
            if not G.has_edge(subj_text, obj_text):
                G.add_edge(subj_text, obj_text, 
                          weight=0, 
                          predicates="", 
                          relations="")
            
            # Append predicates as comma-separated string
            current_predicates = G[subj_text][obj_text]['predicates']
            if current_predicates:
                G[subj_text][obj_text]['predicates'] = current_predicates + "," + predicate
            else:
                G[subj_text][obj_text]['predicates'] = predicate
            
            # Store relation count instead of full relation objects
            current_relations = G[subj_text][obj_text]['relations']
            if current_relations:
                relation_count = int(current_relations.split(',')[0]) + 1
            else:
                relation_count = 1
            G[subj_text][obj_text]['relations'] = f"{relation_count},confidence:{confidence}"
        
        # Set final edge weights
        for (u, v), weight in edge_weights.items():
            if G.has_edge(u, v):
                G[u][v]['weight'] = weight
        
        # Add node attributes for bipartite structure
        for node in gene_nodes:
            if node in G:
                G.nodes[node]['bipartite'] = 0  # Gene side
                G.nodes[node]['node_type'] = 'gene'
        
        for node in function_nodes:
            if node in G:
                G.nodes[node]['bipartite'] = 1  # Function side
                G.nodes[node]['node_type'] = 'function'
        
        # Filter edges by minimum weight
        edges_to_remove = []
        for u, v, data in G.edges(data=True):
            if data['weight'] < self.min_edge_weight:
                edges_to_remove.append((u, v))
        
        G.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        logger.info(f"Gene nodes: {len(gene_nodes & set(G.nodes()))}")
        logger.info(f"Function nodes: {len(function_nodes & set(G.nodes()))}")
        
        self.graph = G
        return G
    
    def project_gene_graph(self, graph: nx.Graph = None) -> nx.Graph:
        """
        Project bipartite graph to gene-gene similarity graph.
        
        Args:
            graph: Bipartite graph (uses self.graph if None)
            
        Returns:
            Gene projection graph
        """
        if graph is None:
            graph = self.graph
        
        if graph is None:
            raise ValueError("No graph available. Build graph first.")
        
        # Get gene nodes (bipartite=0)
        gene_nodes = {n for n, d in graph.nodes(data=True) if d.get('bipartite') == 0}
        
        # Project onto gene nodes
        gene_graph = nx.bipartite.projected_graph(graph, gene_nodes, multigraph=False)
        
        # Calculate edge weights based on shared functions
        for u, v, data in gene_graph.edges(data=True):
            # Get shared functions
            u_functions = set(graph.neighbors(u))
            v_functions = set(graph.neighbors(v))
            shared_functions = u_functions & v_functions
            
            # Weight by number of shared functions and their importance
            weight = 0
            for func in shared_functions:
                func_degree = graph.degree(func)
                weight += 1.0 / (1 + func_degree)  # Rarer functions get higher weight
            
            data['weight'] = weight
            data['shared_functions'] = ",".join(shared_functions) if shared_functions else ""
        
        logger.info(f"Created gene projection with {gene_graph.number_of_nodes()} nodes and {gene_graph.number_of_edges()} edges")
        return gene_graph
    
    def project_function_graph(self, graph: nx.Graph = None) -> nx.Graph:
        """
        Project bipartite graph to function-function similarity graph.
        
        Args:
            graph: Bipartite graph (uses self.graph if None)
            
        Returns:
            Function projection graph
        """
        if graph is None:
            graph = self.graph
        
        if graph is None:
            raise ValueError("No graph available. Build graph first.")
        
        # Get function nodes (bipartite=1)
        function_nodes = {n for n, d in graph.nodes(data=True) if d.get('bipartite') == 1}
        
        # Project onto function nodes
        function_graph = nx.bipartite.projected_graph(graph, function_nodes, multigraph=False)
        
        # Calculate edge weights based on shared genes
        for u, v, data in function_graph.edges(data=True):
            u_genes = set(graph.neighbors(u))
            v_genes = set(graph.neighbors(v))
            shared_genes = u_genes & v_genes
            
            # Weight by number of shared genes
            weight = len(shared_genes)
            data['weight'] = weight
            data['shared_genes'] = ",".join(shared_genes) if shared_genes else ""
        
        logger.info(f"Created function projection with {function_graph.number_of_nodes()} nodes and {function_graph.number_of_edges()} edges")
        return function_graph
    
    def save_graph(self, graph: nx.Graph, filename: str, format: str = "both") -> List[Path]:
        """
        Save graph to disk.
        
        Args:
            graph: NetworkX graph to save
            filename: Base filename (without extension)
            format: Save format ('graphml', 'pickle', or 'both')
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        if format in ['graphml', 'both']:
            graphml_path = self.output_dir / f"{filename}.graphml"
            nx.write_graphml(graph, graphml_path)
            saved_paths.append(graphml_path)
            logger.info(f"Saved graph to {graphml_path}")
        
        if format in ['pickle', 'both']:
            pickle_path = self.output_dir / f"{filename}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(graph, f)
            saved_paths.append(pickle_path)
            logger.info(f"Saved graph to {pickle_path}")
        
        return saved_paths
    
    def load_graph(self, filename: str, format: str = "pickle") -> nx.Graph:
        """
        Load graph from disk.
        
        Args:
            filename: Base filename (with or without extension)
            format: Load format ('graphml' or 'pickle')
            
        Returns:
            Loaded NetworkX graph
        """
        if format == "graphml":
            if not filename.endswith('.graphml'):
                filename += '.graphml'
            path = self.output_dir / filename
            graph = nx.read_graphml(path)
        
        elif format == "pickle":
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            path = self.output_dir / filename
            with open(path, 'rb') as f:
                graph = pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded graph from {path}")
        return graph


class GraphAnalyzer:
    """Analyze knowledge graphs and compute statistics."""
    
    def __init__(self, output_dir: str = "./data/graphs"):
        """
        Initialize graph analyzer.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.community_resolution = float(os.getenv("COMMUNITY_RESOLUTION", 1.0))
    
    def compute_basic_stats(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Compute basic graph statistics.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_connected': nx.is_connected(graph),
            'num_components': nx.number_connected_components(graph),
        }
        
        if stats['num_nodes'] > 0:
            # Degree statistics
            degrees = dict(graph.degree())
            stats['avg_degree'] = np.mean(list(degrees.values()))
            stats['max_degree'] = max(degrees.values())
            stats['min_degree'] = min(degrees.values())
            
            # Check if bipartite
            try:
                stats['is_bipartite'] = nx.is_bipartite(graph)
                if stats['is_bipartite']:
                    gene_nodes = {n for n, d in graph.nodes(data=True) if d.get('bipartite') == 0}
                    function_nodes = {n for n, d in graph.nodes(data=True) if d.get('bipartite') == 1}
                    stats['num_gene_nodes'] = len(gene_nodes)
                    stats['num_function_nodes'] = len(function_nodes)
            except:
                stats['is_bipartite'] = False
            
            # Clustering coefficient (for non-bipartite graphs)
            if not stats.get('is_bipartite', False):
                stats['avg_clustering'] = nx.average_clustering(graph)
            
            # Component sizes
            if stats['num_components'] > 1:
                component_sizes = [len(c) for c in nx.connected_components(graph)]
                stats['largest_component_size'] = max(component_sizes)
                stats['component_sizes'] = component_sizes
        
        return stats
    
    def find_top_hubs(self, graph: nx.Graph, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Find top hub nodes by various centrality measures.
        
        Args:
            graph: NetworkX graph
            top_k: Number of top hubs to return
            
        Returns:
            List of (node, centrality_score) tuples
        """
        centrality_measures = {}
        
        # Degree centrality
        centrality_measures['degree'] = nx.degree_centrality(graph)
        
        # Betweenness centrality (for smaller graphs)
        if graph.number_of_nodes() < 5000:
            centrality_measures['betweenness'] = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
        
        # Eigenvector centrality
        try:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            logger.warning("Could not compute eigenvector centrality")
        
        # PageRank
        centrality_measures['pagerank'] = nx.pagerank(graph)
        
        # Combine centrality scores
        combined_scores = defaultdict(float)
        for measure, scores in centrality_measures.items():
            for node, score in scores.items():
                combined_scores[node] += score
        
        # Normalize by number of measures
        num_measures = len(centrality_measures)
        for node in combined_scores:
            combined_scores[node] /= num_measures
        
        # Return top hubs
        top_hubs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        logger.info(f"Found top {len(top_hubs)} hubs")
        return top_hubs
    
    def detect_communities(self, graph: nx.Graph, method: str = "louvain") -> Dict[int, List[str]]:
        """
        Detect communities in the graph.
        
        Args:
            graph: NetworkX graph
            method: Community detection method ('louvain', 'spectral', 'modularity')
            
        Returns:
            Dictionary mapping community ID to list of nodes
        """
        if graph.number_of_nodes() == 0:
            return {}
        
        communities = {}
        
        if method == "louvain":
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(graph, resolution=self.community_resolution)
                
                # Group nodes by community
                for node, comm_id in partition.items():
                    if comm_id not in communities:
                        communities[comm_id] = []
                    communities[comm_id].append(node)
                    
            except ImportError:
                logger.warning("python-louvain not available, falling back to spectral clustering")
                method = "spectral"
        
        if method == "spectral":
            # Use spectral clustering for smaller graphs
            if graph.number_of_nodes() > 1000:
                logger.warning("Graph too large for spectral clustering, using simple method")
                return self._simple_community_detection(graph)
            
            try:
                # Create adjacency matrix
                adj_matrix = nx.adjacency_matrix(graph)
                
                # Estimate number of communities
                n_communities = min(10, max(2, graph.number_of_nodes() // 100))
                
                # Spectral clustering
                clustering = SpectralClustering(
                    n_clusters=n_communities,
                    affinity='precomputed',
                    random_state=42
                )
                
                labels = clustering.fit_predict(adj_matrix)
                
                # Group nodes by community
                nodes = list(graph.nodes())
                for i, label in enumerate(labels):
                    if label not in communities:
                        communities[label] = []
                    communities[label].append(nodes[i])
                    
            except Exception as e:
                logger.warning(f"Spectral clustering failed: {e}")
                return self._simple_community_detection(graph)
        
        logger.info(f"Found {len(communities)} communities using {method}")
        return communities
    
    def _simple_community_detection(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Simple community detection based on connected components."""
        communities = {}
        for i, component in enumerate(nx.connected_components(graph)):
            communities[i] = list(component)
        return communities
    
    def analyze_communities(self, graph: nx.Graph, communities: Dict[int, List[str]]) -> pd.DataFrame:
        """
        Analyze detected communities.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary of communities
            
        Returns:
            DataFrame with community statistics
        """
        community_stats = []
        
        for comm_id, nodes in communities.items():
            if len(nodes) < 2:
                continue
            
            # Create subgraph for community
            subgraph = graph.subgraph(nodes)
            
            stats = {
                'community_id': comm_id,
                'size': len(nodes),
                'edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph),
                'avg_degree': np.mean([subgraph.degree(n) for n in nodes])
            }
            
            # Bipartite analysis if applicable
            if nx.is_bipartite(subgraph):
                try:
                    gene_nodes = {n for n in nodes if graph.nodes[n].get('bipartite') == 0}
                    function_nodes = {n for n in nodes if graph.nodes[n].get('bipartite') == 1}
                    stats['num_genes'] = len(gene_nodes)
                    stats['num_functions'] = len(function_nodes)
                except:
                    pass
            
            # Most connected nodes in community
            degrees = dict(subgraph.degree())
            if degrees:
                top_node = max(degrees.items(), key=lambda x: x[1])
                stats['top_node'] = top_node[0]
                stats['top_node_degree'] = top_node[1]
            
            community_stats.append(stats)
        
        return pd.DataFrame(community_stats)
    
    def generate_summary_report(self, graph: nx.Graph, 
                              output_file: str = "graph_summary.txt") -> Path:
        """
        Generate a comprehensive summary report.
        
        Args:
            graph: NetworkX graph to analyze
            output_file: Output filename
            
        Returns:
            Path to generated report
        """
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("Gene-RIF Knowledge Graph Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            stats = self.compute_basic_stats(graph)
            f.write("Basic Statistics:\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Top hubs
            f.write("Top Hub Nodes:\n")
            f.write("-" * 15 + "\n")
            top_hubs = self.find_top_hubs(graph)
            for i, (node, score) in enumerate(top_hubs[:10], 1):
                f.write(f"{i:2d}. {node}: {score:.4f}\n")
            f.write("\n")
            
            # Community analysis
            f.write("Community Analysis:\n")
            f.write("-" * 19 + "\n")
            communities = self.detect_communities(graph)
            comm_stats = self.analyze_communities(graph, communities)
            
            if not comm_stats.empty:
                f.write(f"Number of communities: {len(communities)}\n")
                f.write(f"Average community size: {comm_stats['size'].mean():.2f}\n")
                f.write(f"Largest community size: {comm_stats['size'].max()}\n")
                f.write(f"Smallest community size: {comm_stats['size'].min()}\n")
                
                # Top communities by size
                f.write("\nTop 5 communities by size:\n")
                top_communities = comm_stats.nlargest(5, 'size')
                for _, row in top_communities.iterrows():
                    f.write(f"Community {row['community_id']}: {row['size']} nodes, "
                           f"{row['edges']} edges, density={row['density']:.3f}\n")
        
        logger.info(f"Generated summary report: {report_path}")
        return report_path
    
    def export_community_csv(self, graph: nx.Graph, communities: Dict[int, List[str]], 
                            filename: str = "communities.csv") -> Path:
        """
        Export community assignments to CSV.
        
        Args:
            graph: NetworkX graph
            communities: Dictionary of communities
            filename: Output filename
            
        Returns:
            Path to generated CSV file
        """
        csv_path = self.output_dir / filename
        
        # Create rows for CSV
        rows = []
        for comm_id, nodes in communities.items():
            for node in nodes:
                node_data = graph.nodes.get(node, {})
                rows.append({
                    'node': node,
                    'community_id': comm_id,
                    'node_type': node_data.get('node_type', 'unknown'),
                    'degree': graph.degree(node),
                    'bipartite': node_data.get('bipartite', -1)
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Exported community assignments to {csv_path}")
        return csv_path
