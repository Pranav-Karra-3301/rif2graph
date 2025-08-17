#!/bin/bash

# Gene-RIF to Graph Pipeline Update Script
# Suitable for GitHub Actions or cron jobs

set -e  # Exit on any error

echo "==========================================="
echo "Gene-RIF to Graph Pipeline Update"
echo "==========================================="

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
SPECIES="${SPECIES:-human}"
MAX_TEXTS="${MAX_TEXTS:-}"
DEVICE="${DEVICE:-auto}"

# Create directories
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed" "$DATA_DIR/graphs"

echo "Starting pipeline update at $(date)"

# Step 1: Download latest GeneRIF data
echo "Step 1: Downloading GeneRIF data..."
python3 -m gene_rif_graph.scripts.download_data \
    --species "$SPECIES" \
    --data-dir "$DATA_DIR/raw" \
    --fetch-abstracts \
    --species-filter 9606 \
    --force

# Check if download was successful
if [ ! -f "$DATA_DIR/processed/generifs_processed.csv" ]; then
    echo "Error: GeneRIF download failed"
    exit 1
fi

echo "✓ Downloaded and processed GeneRIF data"

# Step 2: Extract entities and relations
echo "Step 2: Extracting biomedical entities and relations..."
EXTRACT_ARGS="--input-file $DATA_DIR/processed/generifs_processed.csv --output-dir $DATA_DIR/processed --device $DEVICE --normalize-genes --normalize-concepts"

# Add max-texts limit if specified (useful for testing)
if [ -n "$MAX_TEXTS" ]; then
    echo "⚠️  Processing limited to $MAX_TEXTS texts for testing"
    EXTRACT_ARGS="$EXTRACT_ARGS --max-texts $MAX_TEXTS"
else
    echo "📄 Processing full dataset (no MAX_TEXTS limit set)"
fi

python3 -m gene_rif_graph.scripts.extract_triplets $EXTRACT_ARGS

# Check if extraction was successful
if [ ! -f "$DATA_DIR/processed/relations.csv" ]; then
    echo "Error: Relation extraction failed"
    exit 1
fi

echo "✓ Extracted entities and relations"

# Step 3: Build knowledge graph
echo "Step 3: Building knowledge graph..."
python3 -m gene_rif_graph.scripts.build_graph \
    --relations-file "$DATA_DIR/processed/relations.csv" \
    --output-dir "$DATA_DIR/graphs" \
    --create-projections \
    --analyze \
    --summary-report

# Check if graph building was successful
if [ ! -f "$DATA_DIR/graphs/bipartite_graph.pkl" ]; then
    echo "Error: Graph building failed"
    exit 1
fi

echo "✓ Built knowledge graphs"

# Step 4: Generate analysis reports
echo "Step 4: Generating analysis reports..."
python3 -m gene_rif_graph.scripts.analyze_graph \
    --graph-file "$DATA_DIR/graphs/bipartite_graph.pkl" \
    --output-dir "$DATA_DIR/graphs" \
    --generate-report \
    --export-csv \
    --analyze-projections

echo "✓ Generated analysis reports"

# Print summary statistics
echo ""
echo "Pipeline Update Summary:"
echo "========================"

# Count files and sizes
if [ -f "$DATA_DIR/processed/generifs_processed.csv" ]; then
    GENERIF_COUNT=$(tail -n +2 "$DATA_DIR/processed/generifs_processed.csv" | wc -l)
    echo "GeneRIFs processed: $GENERIF_COUNT"
fi

if [ -f "$DATA_DIR/processed/relations.csv" ]; then
    RELATION_COUNT=$(tail -n +2 "$DATA_DIR/processed/relations.csv" | wc -l)
    echo "Relations extracted: $RELATION_COUNT"
fi

if [ -f "$DATA_DIR/graphs/graph_summary.txt" ]; then
    echo "Graph statistics:"
    grep -E "(nodes|edges|communities)" "$DATA_DIR/graphs/graph_summary.txt" | head -5
fi

echo ""
echo "Output files:"
echo "-------------"
ls -lh "$DATA_DIR"/processed/*.csv "$DATA_DIR"/graphs/*.{pkl,graphml,csv,txt} 2>/dev/null || true

echo ""
echo "Pipeline completed successfully at $(date)"
echo "Total runtime: $SECONDS seconds"
