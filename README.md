# Gene-RIF to Graph Pipeline

[![Tests](https://github.com/Pranav-Karra-3301/rif2graph/actions/workflows/tests.yml/badge.svg)](https://github.com/Pranav-Karra-3301/rif2graph/actions/workflows/tests.yml)

A comprehensive bioinformatics pipeline that converts GeneRIFs (Gene References Into Function) to knowledge graphs using biomedical NLP and graph analysis techniques.

## 🎯 Project Overview

This project transforms unstructured biomedical text from NCBI GeneRIFs into structured knowledge graphs, enabling large-scale analysis of gene-function relationships and biological pathways.

### Key Features

- **Data Ingestion**: Pull latest GeneRIF dumps from NCBI FTP
- **NLP Pipeline**: Extract entities and relations using SciSpaCy and BioBERT
- **Graph Construction**: Build bipartite gene-function networks with NetworkX
- **Analysis Tools**: Community detection, hub identification, and statistical analysis
- **Automated Updates**: Scheduled pipeline refresh via GitHub Actions

## 🏗 Architecture

```
Raw GeneRIFs → NLP Processing → Relation Extraction → Knowledge Graph → Analysis & Insights
```

### Must-Have Features

✅ **Data Ingestion**
- Pull latest human GeneRIF dump from NCBI FTP
- Optional: fetch linked PubMed abstracts for each GeneRIF

✅ **NLP Pipeline**
- NER → BERN or SciSpaCy biomedical model
- Relation Extraction → BioBERT (fine-tuned) or REBEL
- Post-processing filters:
  - Drop gene–gene or "gene as object" edges
  - Normalize gene IDs (Entrez → official symbol)
  - Normalize object concepts (UMLS/MeSH if available)

✅ **Graph Builder**
- Bipartite Gene ↔ Function NetworkX graph saved to disk (graphml + pickle)
- Utility to project and compute top hubs / Louvain communities

✅ **CLI Scripts** (no notebooks in src)
- `download_data.py` – fetch & cache GeneRIFs
- `extract_triplets.py` – run full NLP pipeline
- `build_graph.py` – turn triples → graph; output summary stats
- `analyze_graph.py` – query hubs / clusters; print CSV reports

✅ **Periodic Refresh**
- `update.sh` bash runner suitable for GitHub Actions or cron

## 📁 Directory Structure

```
gene-rif-graph/
│
├─ .env.example          # API keys / rate-limit tokens
├─ pyproject.toml        # project metadata & dependencies
├─ requirements.txt      # lockfile dependencies
├─ README.md             # this documentation
│
├─ data/                 # cached downloads & intermediate files
│   ├─ raw/
│   ├─ processed/
│   └─ graphs/
│
├─ src/
│   ├─ gene_rif_graph/   # importable package
│   │   ├─ __init__.py
│   │   ├─ ingest.py     # GeneRIF download & parsing
│   │   ├─ nlp.py        # NER + RE helpers
│   │   ├─ filters.py    # post-processing filters
│   │   ├─ normalize.py  # gene/concept normalization
│   │   ├─ graph.py      # graph construction & analysis
│   │   └─ scripts/      # CLI entrypoints
│   │       ├─ download_data.py
│   │       ├─ extract_triplets.py
│   │       ├─ build_graph.py
│   │       └─ analyze_graph.py
│
├─ tests/                # pytest unit tests
│
├─ .github/workflows/    # CI/CD automation
│
└─ update.sh             # one-command refresh
```

## 🛠 Setup & Installation

### Prerequisites

- Python ≥ 3.11
- `uv` (Rust-based pip replacement)

### Quick Start

1. **Install uv**:
```bash
curl -LsSf https://astro.github.io/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone https://github.com/Pranav-Karra-3301/rif2graph.git
cd rif2graph
```

3. **Create environment**:
```bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies**:
```bash
uv pip install -r requirements.txt
```

5. **Install NLP models**:
```bash
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
```

6. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your NCBI email and API key
```

## 🚀 Usage

### Quick Pipeline Run

```bash
# Full pipeline (downloads data, extracts relations, builds graphs)
./update.sh
```

### Step-by-Step Usage

1. **Download GeneRIF data**:
```bash
python -m gene_rif_graph.scripts.download_data \
    --species human \
    --fetch-abstracts \
    --species-filter 9606
```

2. **Extract entities and relations**:
```bash
python -m gene_rif_graph.scripts.extract_triplets \
    --input-file ./data/processed/generifs_processed.csv \
    --normalize-genes \
    --normalize-concepts
```

3. **Build knowledge graph**:
```bash
python -m gene_rif_graph.scripts.build_graph \
    --relations-file ./data/processed/relations.csv \
    --create-projections \
    --analyze
```

4. **Analyze graph**:
```bash
python -m gene_rif_graph.scripts.analyze_graph \
    --graph-file ./data/graphs/bipartite_graph.pkl \
    --generate-report \
    --export-csv
```

### Advanced Usage

**Query specific nodes**:
```bash
python -m gene_rif_graph.scripts.analyze_graph \
    --graph-file ./data/graphs/bipartite_graph.pkl \
    --query-node "TP53"
```

**Analyze communities**:
```bash
python -m gene_rif_graph.scripts.analyze_graph \
    --graph-file ./data/graphs/bipartite_graph.pkl \
    --query-community 0
```

**Custom processing limits**:
```bash
python -m gene_rif_graph.scripts.extract_triplets \
    --input-file ./data/processed/generifs_processed.csv \
    --max-texts 1000 \
    --device cuda
```

## 📊 Output Files

The pipeline generates several output files:

### Processed Data
- `data/processed/generifs_processed.csv` - Cleaned GeneRIF data
- `data/processed/relations.csv` - Extracted relation triplets
- `data/processed/entities.pkl` - Raw entity extractions

### Knowledge Graphs
- `data/graphs/bipartite_graph.pkl` - Main bipartite gene-function graph
- `data/graphs/gene_projection.pkl` - Gene-gene similarity graph
- `data/graphs/function_projection.pkl` - Function-function similarity graph

### Analysis Reports
- `data/graphs/graph_summary.txt` - Comprehensive statistics
- `data/graphs/top_hubs.csv` - Most central nodes
- `data/graphs/communities.csv` - Community assignments
- `data/graphs/edges.csv` - Complete edge list with weights

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=optional_key_for_higher_rate

# Optional
SEMANTIC_SCHOLAR_KEY=optional
CUDA_VISIBLE_DEVICES=0

# Processing settings
MAX_WORKERS=4
BATCH_SIZE=1000
MIN_EDGE_WEIGHT=1
MAX_NODES=50000
```

### Model Configuration

- **NER Model**: `en_ner_bc5cdr_md` (SciSpaCy biomedical NER)
- **RE Model**: `dmis-lab/biobert-base-cased-v1.1` (BioBERT for relations)
- **Device**: Auto-detection (CUDA if available, else CPU)

## 🧪 Testing

Run the test suite:

```bash
# Install dev dependencies
uv pip install -r requirements-dev.txt

# Run tests with coverage
pytest tests/ --cov=src/gene_rif_graph --cov-report=term-missing

# Run specific test modules
pytest tests/test_nlp.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/gene_rif_graph/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 📈 Performance & Scalability

### Typical Processing Times

- **GeneRIF Download**: ~5-10 minutes
- **NLP Processing**: ~30-60 minutes (10K GeneRIFs)
- **Graph Construction**: ~5-10 minutes
- **Analysis**: ~2-5 minutes

### Memory Requirements

- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM (for full human dataset)
- **GPU**: Optional but recommended for large-scale processing

### Scaling Options

- **Batch Processing**: Adjust `BATCH_SIZE` environment variable
- **Parallel Processing**: Set `MAX_WORKERS` for concurrent operations
- **Memory Optimization**: Use `MAX_TEXTS` for incremental processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies (`uv pip install -r requirements-dev.txt`)
4. Make changes and add tests
5. Run tests and quality checks
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Use type hints for all functions
- Maintain >80% test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NCBI**: For providing GeneRIF data
- **SciSpaCy**: Biomedical NLP models
- **BioBERT**: Biomedical language representations
- **NetworkX**: Graph analysis library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Pranav-Karra-3301/rif2graph/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Pranav-Karra-3301/rif2graph/discussions)
- **Email**: [your_email@example.com](mailto:your_email@example.com)

## 🔄 Automated Updates

The pipeline includes automated weekly updates via GitHub Actions:

- **Schedule**: Every Sunday at 2 AM UTC
- **Manual Trigger**: Available via GitHub Actions UI
- **Artifacts**: Generated graphs and reports
- **Releases**: Automated releases for major updates

Monitor the pipeline status: [![Update Graph](https://github.com/Pranav-Karra-3301/rif2graph/actions/workflows/update-graph.yml/badge.svg)](https://github.com/Pranav-Karra-3301/rif2graph/actions/workflows/update-graph.yml)
