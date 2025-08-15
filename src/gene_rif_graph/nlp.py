"""
NLP pipeline for biomedical named entity recognition and relation extraction.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import spacy
import scispacy
from scispacy.linking import EntityLinker
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BioNERExtractor:
    """Biomedical Named Entity Recognition using SciSpaCy."""
    
    def __init__(self, model_name: str = "en_ner_bc5cdr_md"):
        """
        Initialize NER extractor.
        
        Args:
            model_name: SciSpaCy model name (e.g., 'en_ner_bc5cdr_md', 'en_ner_bionlp13cg_md')
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the SciSpaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded NER model: {self.model_name}")
            
            # Add entity linker for UMLS concept IDs
            if "umls" not in self.nlp.pipe_names:
                self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                
        except OSError:
            logger.error(f"Model {self.model_name} not found. Install with: pip install {self.model_name}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract biomedical entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with keys: text, label, start, end, umls_id
        """
        if not self.nlp:
            raise RuntimeError("NER model not loaded")
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_dict = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'umls_id': None
            }
            
            # Add UMLS concept ID if available
            if hasattr(ent._, 'umls_ents') and ent._.umls_ents:
                # Take the highest scoring UMLS concept
                best_concept = max(ent._.umls_ents, key=lambda x: x[1])
                entity_dict['umls_id'] = best_concept[0]
            
            entities.append(entity_dict)
        
        return entities
    
    def batch_extract_entities(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities from multiple texts."""
        if not self.nlp:
            raise RuntimeError("NER model not loaded")
        
        results = []
        for doc in tqdm(self.nlp.pipe(texts, batch_size=50), total=len(texts), desc="Extracting entities"):
            entities = []
            for ent in doc.ents:
                entity_dict = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'umls_id': None
                }
                
                if hasattr(ent._, 'umls_ents') and ent._.umls_ents:
                    best_concept = max(ent._.umls_ents, key=lambda x: x[1])
                    entity_dict['umls_id'] = best_concept[0]
                
                entities.append(entity_dict)
            
            results.append(entities)
        
        return results


class RelationExtractor:
    """Biomedical relation extraction using BioBERT or REBEL."""
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", 
                 device: str = "auto"):
        """
        Initialize relation extractor.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load the relation extraction model."""
        try:
            # For BioBERT-based relation extraction
            if "biobert" in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info(f"Loaded BioBERT model: {self.model_name} on {self.device}")
            
            # For REBEL or other seq2seq models
            elif "rebel" in self.model_name.lower():
                self.pipeline = pipeline(
                    "text2text-generation", 
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info(f"Loaded REBEL model: {self.model_name} on {self.device}")
            
            else:
                # Generic relation extraction pipeline
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info(f"Loaded RE model: {self.model_name} on {self.device}")
                
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations between entities in text.
        
        Args:
            text: Input text
            entities: List of entities from NER
            
        Returns:
            List of relation dictionaries
        """
        relations = []
        
        # Generate entity pairs
        entity_pairs = []
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], i+1):
                if ent1['label'] != ent2['label']:  # Different entity types
                    entity_pairs.append((ent1, ent2))
        
        # Extract relations for each pair
        for ent1, ent2 in entity_pairs:
            relation = self._extract_relation_pair(text, ent1, ent2)
            if relation:
                relations.append(relation)
        
        return relations
    
    def _extract_relation_pair(self, text: str, ent1: Dict, ent2: Dict) -> Optional[Dict[str, Any]]:
        """Extract relation between a pair of entities."""
        try:
            # Create context window around entities
            start_pos = min(ent1['start'], ent2['start'])
            end_pos = max(ent1['end'], ent2['end'])
            
            # Expand context
            context_start = max(0, start_pos - 100)
            context_end = min(len(text), end_pos + 100)
            context = text[context_start:context_end]
            
            # Mark entities in context
            marked_text = self._mark_entities_in_text(context, ent1, ent2, context_start)
            
            if self.pipeline and "rebel" in self.model_name.lower():
                # Use REBEL for relation extraction
                result = self.pipeline(marked_text, max_length=512, num_return_sequences=1)
                if result and result[0]['generated_text']:
                    return self._parse_rebel_output(result[0]['generated_text'], ent1, ent2)
            
            else:
                # Use BioBERT or other models for relation classification
                # This would require a fine-tuned model for relation classification
                # For now, we'll use a simple heuristic-based approach
                return self._heuristic_relation_extraction(text, ent1, ent2)
            
        except Exception as e:
            logger.warning(f"Error extracting relation between {ent1['text']} and {ent2['text']}: {e}")
            return None
    
    def _mark_entities_in_text(self, text: str, ent1: Dict, ent2: Dict, offset: int) -> str:
        """Mark entities in text with special tokens."""
        # Adjust entity positions relative to context
        ent1_start = ent1['start'] - offset
        ent1_end = ent1['end'] - offset
        ent2_start = ent2['start'] - offset
        ent2_end = ent2['end'] - offset
        
        # Insert markers (process in reverse order to maintain positions)
        markers = [
            (max(ent1_end, ent2_end), "</ent>"),
            (max(ent1_start, ent2_start), "<ent>"),
            (min(ent1_end, ent2_end), "</ent>"),
            (min(ent1_start, ent2_start), "<ent>")
        ]
        
        for pos, marker in sorted(markers, reverse=True):
            if 0 <= pos <= len(text):
                text = text[:pos] + marker + text[pos:]
        
        return text
    
    def _parse_rebel_output(self, output: str, ent1: Dict, ent2: Dict) -> Optional[Dict[str, Any]]:
        """Parse REBEL model output to extract relations."""
        # REBEL outputs relations in a specific format
        # This is a simplified parser - would need refinement for production
        try:
            if "<triplet>" in output:
                triplets = output.split("<triplet>")[1:]
                for triplet in triplets:
                    parts = triplet.split("<subj>")[1].split("<obj>")
                    if len(parts) >= 2:
                        subj_rel = parts[0].split("<pred>")
                        if len(subj_rel) >= 2:
                            subject = subj_rel[0].strip()
                            predicate = subj_rel[1].strip()
                            object_text = parts[1].strip()
                            
                            # Check if this triplet matches our entities
                            if (subject.lower() in ent1['text'].lower() or 
                                subject.lower() in ent2['text'].lower()):
                                return {
                                    'subject': ent1,
                                    'predicate': predicate,
                                    'object': ent2,
                                    'confidence': 0.8  # Default confidence
                                }
        except Exception as e:
            logger.warning(f"Error parsing REBEL output: {e}")
        
        return None
    
    def _heuristic_relation_extraction(self, text: str, ent1: Dict, ent2: Dict) -> Optional[Dict[str, Any]]:
        """Simple heuristic-based relation extraction."""
        # Define common biomedical relation patterns
        gene_labels = {'GENE', 'PROTEIN'}
        disease_labels = {'DISEASE', 'DISORDER'}
        chemical_labels = {'CHEMICAL', 'DRUG'}
        
        ent1_label = ent1['label']
        ent2_label = ent2['label']
        
        # Gene-Disease relations
        if ((ent1_label in gene_labels and ent2_label in disease_labels) or
            (ent1_label in disease_labels and ent2_label in gene_labels)):
            
            subject = ent1 if ent1_label in gene_labels else ent2
            object_ent = ent2 if ent1_label in gene_labels else ent1
            
            # Look for relation keywords in context
            context = text[min(ent1['start'], ent2['start']):max(ent1['end'], ent2['end'])]
            context_lower = context.lower()
            
            if any(word in context_lower for word in ['associated', 'linked', 'implicated']):
                predicate = 'associated_with'
            elif any(word in context_lower for word in ['causes', 'leads to', 'results in']):
                predicate = 'causes'
            elif any(word in context_lower for word in ['treats', 'therapy', 'treatment']):
                predicate = 'treats'
            else:
                predicate = 'related_to'
            
            return {
                'subject': subject,
                'predicate': predicate,
                'object': object_ent,
                'confidence': 0.6  # Lower confidence for heuristic
            }
        
        # Chemical-Disease relations
        elif ((ent1_label in chemical_labels and ent2_label in disease_labels) or
              (ent1_label in disease_labels and ent2_label in chemical_labels)):
            
            subject = ent1 if ent1_label in chemical_labels else ent2
            object_ent = ent2 if ent1_label in chemical_labels else ent1
            
            return {
                'subject': subject,
                'predicate': 'treats',
                'object': object_ent,
                'confidence': 0.5
            }
        
        return None
    
    def batch_extract_relations(self, texts: List[str], 
                              entities_list: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Extract relations from multiple texts."""
        relations_list = []
        
        for text, entities in tqdm(zip(texts, entities_list), 
                                  total=len(texts), desc="Extracting relations"):
            relations = self.extract_relations(text, entities)
            relations_list.append(relations)
        
        return relations_list
