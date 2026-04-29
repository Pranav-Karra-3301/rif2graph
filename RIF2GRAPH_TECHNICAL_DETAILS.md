# RIF2GRAPH Pipeline: Technical Details

## Overview

The rif2graph pipeline reads plain-text scientific sentences about genes (GeneRIFs) and turns them into a structured knowledge graph. This document explains each step of the process, what technologies are used, and what the current limitations are.

---

## The Pipeline: A Step-by-Step Guide

The process can be broken down into six main steps:

### Step 1: Data Ingestion

- **What it does:** Downloads the raw data from the NCBI database.
- **Input:** A large file named `generifs_basic.gz` containing millions of GeneRIF entries.
- **Details:** Each entry includes a gene ID, the scientific text (the "RIF"), and a reference to the publication (PubMed ID). We focus on human genes (`tax_id: 9606`).

### Step 2: Finding Biological Entities (Named Entity Recognition)

- **What it does:** Scans the text to identify and label important biological terms.
- **Technology:** `SciSpacy`, a library for biomedical text processing.
- **Details:**
    - **Entity Types:** It finds Genes/Proteins, Diseases, Chemicals, and Biological Processes.
    - **UMLS Linking:** It links these entities to a standard medical vocabulary (UMLS) to ensure concepts are unique. For example, "heart attack" and "myocardial infarction" are linked to the same UMLS ID.
    - **Regex Fallback:** If SciSpacy misses a gene, a set of regular expressions tries to find gene symbols as a backup.

### Step 3: Extracting Relationships

- **What it does:** Determines how the entities found in Step 2 are related to each other.
- **Technology:** `REBEL`, a transformer-based AI model.
- **Details:**
    - **Triplet Generation:** REBEL reads a sentence and generates structured "triplets" like `<subject> <predicate> <object>`. For example, from "TP53 mutations cause lung cancer," it would extract `<TP53> <causes> <lung cancer>`.
    - **Fallback Methods:** If REBEL fails, the pipeline falls back to simpler methods:
        1.  **Linguistic Patterns:** Looks for keywords between entities (e.g., "is involved in," "regulates").
        2.  **Dependency Parsing:** Analyzes the grammatical structure of the sentence to find connecting verbs.
        3.  **Heuristic Rules:** As a last resort, it creates a default link based on entity types (e.g., a Gene and a Disease are assumed to be `associated_with`).

### Step 4: Cleaning the Data (Filtering)

- **What it does:** Removes low-quality or "noisy" relationships to keep the graph accurate. This is a critical step and also a primary source of the current issues.
- **Details:** Several filters are applied:
    - **Confidence Score:** Each relationship has a confidence score. Any relationship with a score below 0.2 is removed.
    - **Bipartite Structure:** The graph is designed to connect *Genes* to *Concepts* (like diseases or processes). Any relationship that connects two genes directly (`Gene -> Gene`) or two concepts directly (`Disease -> Chemical`) is **removed**.
    - **Blacklist Filter:** Removes uninformative relationships like `<TP53> <is> <gene>`.
    - **Quality Filter:** Removes entities that are too short, too long, or match common noise words like "Figure" or "Table".

### Step 5: Standardizing Terms (Normalization)

- **What it does:** Ensures that the same entity is represented by a single, standard name.
- **Details:**
    - **Gene Normalization:** Maps gene aliases to their official symbol (e.g., "p53" becomes "TP53"). It uses the official NCBI gene database for this.
    - **Concept Normalization:** Cleans up disease and chemical names (e.g., "Alzheimer's Disease" and "Alzheimers" become the same). It also uses a clustering algorithm to group very similar terms that aren't exact matches.

### Step 6: Building the Graph

- **What it does:** Assembles the final, cleaned-up triplets into a network.
- **Technology:** `NetworkX`, a Python library for graph analysis.
- **Details:**
    - **Nodes:** The entities (genes, diseases, etc.).
    - **Edges:** The relationships (predicates) that connect the nodes.
    - **Edge Weights:** The weight of an edge is based on the confidence of the extracted relationship and how many times it appeared in the data.

---

## Current Status & Known Issues

The pipeline is functional, but the analysis shows it is not processing all GeneRIFs effectively, leading to a fragmented knowledge graph.

- **Key Problem:** The graph is not fully connected and is broken into **72 separate components**. This indicates that many genes and their functions are not being linked to the main network.

- **Primary Cause:** The filtering rules, while designed to ensure high quality, are likely too aggressive.
    - The **`Gene -> Gene` filter** is a major contributor. Many GeneRIFs describe a gene's function by relating it to another gene (e.g., "Gene A activates Gene B"). By deleting all such links, we lose the context for Gene A and it may become isolated if it has no other relationships.
    - The strict **bipartite structure enforcement** also removes potentially useful, non-direct relationships, further breaking connections.

### What's Not Working Properly

- **Incomplete Processing of GeneRIFs:** Due to the issues above, a significant number of GeneRIFs are not being successfully converted into meaningful, connected knowledge. The pipeline correctly extracts entities and relationships, but the filtering step discards many of them, leading to the fragmented graph shown in the analysis report.

### Potential Next Steps

1.  **Re-evaluate the Filtering Strategy:** The `Gene -> Gene` filter could be relaxed or replaced with a more nuanced approach. Instead of deleting these relationships, they could be kept but given a lower weight.
2.  **Improve Normalization:** Further refinement of the concept normalization and clustering could help merge more synonymous nodes, which would help connect different parts of the graph.
3.  **Enhance Relation Extraction:** Fine-tuning the REBEL model on a biomedical-specific dataset could improve its accuracy and reduce its failure rate, meaning the pipeline would have to rely less on low-confidence fallback methods.
