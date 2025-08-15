#!/usr/bin/env python3
"""
Demo script showing basic functionality of the gene-rif-graph pipeline.
"""

import os
import tempfile
import pandas as pd
from pathlib import Path

# Set up demo environment
print("🧬 Gene-RIF to Graph Pipeline Demo")
print("=" * 40)

# Create temporary directory for demo
with tempfile.TemporaryDirectory() as temp_dir:
    demo_dir = Path(temp_dir)
    
    # Create sample GeneRIF data
    print("📋 Creating sample GeneRIF data...")
    sample_data = [
        {
            'tax_id': 9606,
            'gene_id': 7157,
            'pmid': 12345,
            'timestamp': '2023-01-01',
            'text': 'TP53 is a tumor suppressor gene that is frequently mutated in human cancers and plays a crucial role in DNA damage response.'
        },
        {
            'tax_id': 9606,
            'gene_id': 672,
            'pmid': 12346,
            'timestamp': '2023-01-02',
            'text': 'BRCA1 mutations are associated with increased risk of breast and ovarian cancer due to defects in DNA repair mechanisms.'
        },
        {
            'tax_id': 9606,
            'gene_id': 1956,
            'pmid': 12347,
            'timestamp': '2023-01-03',
            'text': 'EGFR overexpression promotes cell proliferation and survival in lung cancer through activation of downstream signaling pathways.'
        },
        {
            'tax_id': 9606,
            'gene_id': 5728,
            'pmid': 12348,
            'timestamp': '2023-01-04',
            'text': 'PTEN acts as a tumor suppressor by negatively regulating the PI3K-AKT pathway and controlling cell growth and apoptosis.'
        },
        {
            'tax_id': 9606,
            'gene_id': 2064,
            'pmid': 12349,
            'timestamp': '2023-01-05',
            'text': 'ERBB2 amplification is found in breast cancer and serves as a target for therapeutic interventions like trastuzumab.'
        }
    ]
    
    # Save sample data
    df = pd.DataFrame(sample_data)
    sample_file = demo_dir / "sample_generifs.csv"
    df.to_csv(sample_file, index=False)
    print(f"✅ Created sample data with {len(sample_data)} GeneRIFs")
    
    # Demo NLP processing
    print("\n🔍 Demo: Entity Extraction...")
    try:
        from gene_rif_graph.nlp import BioNERExtractor
        
        # Initialize NER extractor (with fallback)
        try:
            ner = BioNERExtractor(model_name="en_core_web_sm")  # Fallback to basic model
        except:
            print("⚠️  SciSpaCy model not available, using mock extraction...")
            # Mock entity extraction for demo
            entities = [
                [
                    {'text': 'TP53', 'label': 'GENE', 'start': 0, 'end': 4},
                    {'text': 'cancer', 'label': 'DISEASE', 'start': 70, 'end': 76}
                ]
            ]
        else:
            # Extract entities from first text
            entities = ner.batch_extract_entities([sample_data[0]['text']])
        
        print(f"✅ Extracted entities from sample text:")
        if entities and entities[0]:
            for ent in entities[0][:3]:  # Show first 3 entities
                print(f"   - {ent['text']} ({ent['label']})")
        else:
            print("   - TP53 (GENE)")
            print("   - cancer (DISEASE)")
    
    except ImportError:
        print("⚠️  NLP modules not available, showing mock results...")
        print("   - TP53 (GENE)")
        print("   - tumor suppressor (FUNCTION)")
        print("   - cancer (DISEASE)")
    
    # Demo relation extraction
    print("\n🔗 Demo: Relation Extraction...")
    sample_relations = [
        {
            'subject': {'text': 'TP53', 'label': 'GENE'},
            'predicate': 'involved_in',
            'object': {'text': 'DNA damage response', 'label': 'FUNCTION'},
            'confidence': 0.9
        },
        {
            'subject': {'text': 'BRCA1', 'label': 'GENE'},
            'predicate': 'associated_with',
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
    
    print("✅ Extracted relations:")
    for rel in sample_relations:
        print(f"   - {rel['subject']['text']} → {rel['predicate']} → {rel['object']['text']}")
    
    # Demo graph construction
    print("\n📊 Demo: Graph Construction...")
    try:
        from gene_rif_graph.graph import GraphBuilder
        
        builder = GraphBuilder(output_dir=demo_dir)
        graph = builder.build_graph_from_triplets(sample_relations)
        
        print(f"✅ Built bipartite graph:")
        print(f"   - Nodes: {graph.number_of_nodes()}")
        print(f"   - Edges: {graph.number_of_edges()}")
        
        # Show some nodes by type
        gene_nodes = [n for n, d in graph.nodes(data=True) if d.get('bipartite') == 0]
        function_nodes = [n for n, d in graph.nodes(data=True) if d.get('bipartite') == 1]
        
        print(f"   - Gene nodes: {gene_nodes}")
        print(f"   - Function nodes: {function_nodes}")
        
    except ImportError:
        print("⚠️  Graph modules not available, showing mock results...")
        print("   - Nodes: 6")
        print("   - Edges: 3")
        print("   - Gene nodes: ['TP53', 'BRCA1', 'EGFR']")
        print("   - Function nodes: ['DNA damage response', 'breast cancer', 'cell proliferation']")
    
    # Demo analysis
    print("\n📈 Demo: Graph Analysis...")
    try:
        from gene_rif_graph.graph import GraphAnalyzer
        
        analyzer = GraphAnalyzer(output_dir=demo_dir)
        stats = analyzer.compute_basic_stats(graph)
        
        print("✅ Graph statistics:")
        print(f"   - Density: {stats['density']:.3f}")
        print(f"   - Is bipartite: {stats['is_bipartite']}")
        print(f"   - Connected components: {stats['num_components']}")
        
        # Find top hubs
        hubs = analyzer.find_top_hubs(graph, top_k=3)
        print("✅ Top hub nodes:")
        for node, score in hubs:
            print(f"   - {node}: {score:.3f}")
    
    except (ImportError, NameError):
        print("⚠️  Analysis modules not available, showing mock results...")
        print("   - Density: 0.500")
        print("   - Is bipartite: True")
        print("   - Connected components: 1")
        print("   - Top hubs: TP53 (0.750), BRCA1 (0.500), EGFR (0.250)")

print("\n🎉 Demo completed!")
print("\n📚 To run the full pipeline:")
print("   1. Install dependencies: uv pip install -r requirements.txt")
print("   2. Configure .env file with NCBI credentials")
print("   3. Run: ./update.sh")
print("   4. Or run individual scripts:")
print("      - python -m gene_rif_graph.scripts.download_data")
print("      - python -m gene_rif_graph.scripts.extract_triplets")
print("      - python -m gene_rif_graph.scripts.build_graph")
print("      - python -m gene_rif_graph.scripts.analyze_graph")
print("\n📖 See README.md for detailed usage instructions.")
