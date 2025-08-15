"""Test graph construction and analysis."""

import pytest
import networkx as nx
from gene_rif_graph.graph import GraphBuilder, GraphAnalyzer


class TestGraphBuilder:
    """Test graph construction functionality."""
    
    def test_init(self, temp_data_dir):
        """Test graph builder initialization."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        assert builder.output_dir.exists()
        assert builder.graph is None
    
    def test_build_graph_from_triplets(self, temp_data_dir, sample_relations):
        """Test building bipartite graph from triplets."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Check bipartite structure
        gene_nodes = {n for n, d in graph.nodes(data=True) if d.get('bipartite') == 0}
        function_nodes = {n for n, d in graph.nodes(data=True) if d.get('bipartite') == 1}
        
        assert len(gene_nodes) > 0
        assert len(function_nodes) > 0
        
        # Check that genes and functions are connected
        for gene in gene_nodes:
            neighbors = list(graph.neighbors(gene))
            assert any(neighbor in function_nodes for neighbor in neighbors)
    
    def test_project_gene_graph(self, temp_data_dir, sample_relations):
        """Test gene projection creation."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        bipartite_graph = builder.build_graph_from_triplets(sample_relations)
        
        gene_graph = builder.project_gene_graph(bipartite_graph)
        
        assert isinstance(gene_graph, nx.Graph)
        # Gene projection should have only gene nodes
        for node in gene_graph.nodes():
            assert bipartite_graph.nodes[node].get('bipartite') == 0
    
    def test_project_function_graph(self, temp_data_dir, sample_relations):
        """Test function projection creation."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        bipartite_graph = builder.build_graph_from_triplets(sample_relations)
        
        function_graph = builder.project_function_graph(bipartite_graph)
        
        assert isinstance(function_graph, nx.Graph)
        # Function projection should have only function nodes
        for node in function_graph.nodes():
            assert bipartite_graph.nodes[node].get('bipartite') == 1
    
    def test_save_and_load_graph(self, temp_data_dir, sample_relations):
        """Test saving and loading graphs."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        # Save graph
        saved_paths = builder.save_graph(graph, "test_graph", format="both")
        assert len(saved_paths) == 2
        
        # Load graph
        loaded_graph = builder.load_graph("test_graph", format="pickle")
        
        assert loaded_graph.number_of_nodes() == graph.number_of_nodes()
        assert loaded_graph.number_of_edges() == graph.number_of_edges()


class TestGraphAnalyzer:
    """Test graph analysis functionality."""
    
    def test_init(self, temp_data_dir):
        """Test analyzer initialization."""
        analyzer = GraphAnalyzer(output_dir=temp_data_dir)
        assert analyzer.output_dir.exists()
    
    def test_compute_basic_stats(self, temp_data_dir, sample_relations):
        """Test basic statistics computation."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        analyzer = GraphAnalyzer(output_dir=temp_data_dir)
        stats = analyzer.compute_basic_stats(graph)
        
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'density' in stats
        assert 'is_bipartite' in stats
        assert stats['num_nodes'] > 0
        assert stats['num_edges'] > 0
        assert stats['is_bipartite'] is True
    
    def test_find_top_hubs(self, temp_data_dir, sample_relations):
        """Test hub node identification."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        analyzer = GraphAnalyzer(output_dir=temp_data_dir)
        top_hubs = analyzer.find_top_hubs(graph, top_k=5)
        
        assert isinstance(top_hubs, list)
        assert len(top_hubs) <= 5
        
        if top_hubs:
            assert isinstance(top_hubs[0], tuple)
            assert len(top_hubs[0]) == 2
            # Check that scores are in descending order
            scores = [score for _, score in top_hubs]
            assert scores == sorted(scores, reverse=True)
    
    def test_detect_communities(self, temp_data_dir, sample_relations):
        """Test community detection."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        analyzer = GraphAnalyzer(output_dir=temp_data_dir)
        communities = analyzer.detect_communities(graph, method="spectral")
        
        assert isinstance(communities, dict)
        
        if communities:
            # Check that all nodes are assigned to communities
            all_community_nodes = set()
            for nodes in communities.values():
                all_community_nodes.update(nodes)
            
            assert len(all_community_nodes) <= graph.number_of_nodes()
    
    def test_analyze_communities(self, temp_data_dir, sample_relations):
        """Test community analysis."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        analyzer = GraphAnalyzer(output_dir=temp_data_dir)
        communities = analyzer.detect_communities(graph)
        comm_stats = analyzer.analyze_communities(graph, communities)
        
        assert not comm_stats.empty or len(communities) == 0
        
        if not comm_stats.empty:
            assert 'community_id' in comm_stats.columns
            assert 'size' in comm_stats.columns
            assert 'edges' in comm_stats.columns
    
    def test_generate_summary_report(self, temp_data_dir, sample_relations):
        """Test summary report generation."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        analyzer = GraphAnalyzer(output_dir=temp_data_dir)
        report_path = analyzer.generate_summary_report(graph)
        
        assert report_path.exists()
        assert report_path.suffix == '.txt'
        
        # Check that report contains expected content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Basic Statistics' in content
            assert 'Top Hub Nodes' in content
            assert 'Community Analysis' in content
    
    def test_export_community_csv(self, temp_data_dir, sample_relations):
        """Test community CSV export."""
        builder = GraphBuilder(output_dir=temp_data_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        analyzer = GraphAnalyzer(output_dir=temp_data_dir)
        communities = analyzer.detect_communities(graph)
        csv_path = analyzer.export_community_csv(graph, communities)
        
        assert csv_path.exists()
        assert csv_path.suffix == '.csv'
        
        # Check CSV content
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        expected_columns = ['node', 'community_id', 'node_type', 'degree', 'bipartite']
        for col in expected_columns:
            assert col in df.columns
