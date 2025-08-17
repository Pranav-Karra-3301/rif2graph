"""
NLP pipeline for biomedical named entity recognition and relation extraction.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
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
    
    def __init__(self, model_name: str = "en_ner_bionlp13cg_md"):
        """
        Initialize NER extractor.
        
        Args:
            model_name: SciSpaCy model name (en_ner_bionlp13cg_md for genes, en_ner_bc5cdr_md for chemicals/diseases)
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
        
        # Extract entities from spaCy NER
        for ent in doc.ents:
            entity_dict = {
                'text': ent.text,
                'label': self._normalize_entity_label(ent.label_),
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
        
        # Add regex-based gene entities as fallback
        regex_entities = self._extract_gene_patterns(text)
        entities.extend(regex_entities)
        
        # Remove duplicates (prefer NER over regex)
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _normalize_entity_label(self, label: str) -> str:
        """Normalize entity labels to standard types."""
        label_mapping = {
            'GENE_OR_GENE_PRODUCT': 'GENE',
            'PROTEIN': 'GENE',
            'CANCER': 'DISEASE', 
            'DISEASE_OR_DISORDER': 'DISEASE',
            'ORGANISM': 'SPECIES',
            'CELL_TYPE': 'CELL',
            'CELL_LINE': 'CELL',
            'ORGAN': 'ANATOMY',
            'TISSUE': 'ANATOMY',
            'SIMPLE_CHEMICAL': 'CHEMICAL',
            'CHEMICAL': 'CHEMICAL',
            'DRUG': 'CHEMICAL'
        }
        return label_mapping.get(label, label)
    
    def _extract_gene_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract gene symbols using regex patterns."""
        import re
        
        entities = []
        
        # Common gene symbol patterns
        patterns = [
            # Standard gene symbols (2-10 uppercase letters, may include numbers)
            r'\b[A-Z]{2,}[0-9]*[A-Z]*\b',
            # Gene symbols with numbers/greek letters
            r'\b[A-Z]+[0-9]+[A-Z]*\b',
            r'\b[A-Z]+[α-ω]+[0-9]*\b',
            # Common gene families
            r'\b(TP53|BRCA[12]|EGFR|MYC|RAS|APC|PTEN|ATM|CHEK[12]|MDM[24])\b',
            r'\b(CYP[0-9]+[A-Z][0-9]*|UGT[0-9]+[A-Z][0-9]*|NAT[12]|GSTM?[0-9])\b'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                gene_text = match.group()
                
                # Filter out common false positives
                if not self._is_likely_gene(gene_text):
                    continue
                
                entity = {
                    'text': gene_text,
                    'label': 'GENE',
                    'start': match.start(),
                    'end': match.end(),
                    'umls_id': None
                }
                entities.append(entity)
        
        return entities
    
    def _is_likely_gene(self, text: str) -> bool:
        """Filter out false positive gene matches."""
        # Skip common false positives
        false_positives = {
            'DNA', 'RNA', 'ATP', 'ADP', 'GTP', 'GDP', 'NAD', 'NADH', 'FAD', 'FADH',
            'PCR', 'RT', 'PAGE', 'SDS', 'PBS', 'BSA', 'DMSO', 'EDTA', 'TRIS',
            'AND', 'OR', 'NOT', 'THE', 'FOR', 'WITH', 'FROM', 'INTO', 'THAT',
            'ALL', 'ANY', 'CAN', 'MAY', 'WILL', 'ALSO', 'WERE', 'ARE', 'WAS'
        }
        
        if text.upper() in false_positives:
            return False
        
        # Must be 2-15 characters
        if len(text) < 2 or len(text) > 15:
            return False
        
        # Should contain at least one letter
        if not any(c.isalpha() for c in text):
            return False
        
        return True
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities, preferring NER over regex."""
        seen = set()
        deduplicated = []
        
        # Sort to prioritize NER entities (they don't have 'regex' source)
        entities.sort(key=lambda x: ('regex' in x.get('source', ''), x['start']))
        
        for entity in entities:
            # Create overlap key based on text span
            start, end = entity['start'], entity['end']
            
            # Check for overlaps with existing entities
            overlaps = False
            for seen_start, seen_end in seen:
                if (start < seen_end and end > seen_start):  # Overlap detected
                    overlaps = True
                    break
            
            if not overlaps:
                seen.add((start, end))
                deduplicated.append(entity)
        
        return deduplicated
    
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
    """Biomedical relation extraction using REBEL or pattern-based methods."""
    
    def __init__(self, model_name: str = "Babelscape/rebel-large", 
                 device: str = "auto"):
        """
        Initialize relation extractor.
        
        Args:
            model_name: HuggingFace model name (default: REBEL for relation extraction)
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
            # For REBEL relation extraction (preferred)
            if "rebel" in self.model_name.lower():
                self.pipeline = pipeline(
                    "text2text-generation", 
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    max_length=256,
                    do_sample=False
                )
                logger.info(f"Loaded REBEL model: {self.model_name} on {self.device}")
            
            else:
                # Fallback to pattern-based extraction
                logger.warning(f"Model {self.model_name} not recognized, using pattern-based extraction")
                self.pipeline = None
                
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            logger.warning("Falling back to pattern-based relation extraction")
            self.pipeline = None
    
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
            
            # Expand context but keep it manageable for REBEL
            context_start = max(0, start_pos - 50)
            context_end = min(len(text), end_pos + 50)
            context = text[context_start:context_end]
            
            if self.pipeline:
                # Use REBEL for relation extraction
                try:
                    result = self.pipeline(context, max_length=256, num_return_sequences=1)
                    if result and len(result) > 0:
                        rebel_relation = self._parse_rebel_output(result[0]['generated_text'], ent1, ent2, context)
                        if rebel_relation:
                            return rebel_relation
                except Exception as e:
                    logger.debug(f"REBEL extraction failed: {e}")
            
            # Fallback to pattern-based extraction
            return self._heuristic_relation_extraction(context, ent1, ent2)
            
        except Exception as e:
            logger.debug(f"Error extracting relation between {ent1['text']} and {ent2['text']}: {e}")
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
    
    def _parse_rebel_output(self, output: str, ent1: Dict, ent2: Dict, context: str) -> Optional[Dict[str, Any]]:
        """Parse REBEL model output to extract relations."""
        try:
            # REBEL outputs format: <triplet> subject <subj> predicate <pred> object <obj>
            if "<triplet>" in output:
                triplets = output.split("<triplet>")[1:]
                for triplet in triplets:
                    # Clean up the triplet
                    triplet = triplet.strip()
                    
                    # Parse using regex for more robust extraction
                    import re
                    pattern = r'(.+?)<subj>(.+?)<pred>(.+?)<obj>(.+?)(?=<triplet>|$)'
                    matches = re.findall(pattern, triplet)
                    
                    if matches:
                        _, subject, predicate, object_text = matches[0]
                        subject = subject.strip()
                        predicate = predicate.strip()
                        object_text = object_text.strip()
                        
                        # Match extracted entities to our input entities
                        subj_ent, obj_ent = self._match_entities_to_triplet(
                            subject, object_text, ent1, ent2, context
                        )
                        
                        if subj_ent and obj_ent and predicate:
                            return {
                                'subject': subj_ent,
                                'predicate': predicate.replace(' ', '_').lower(),
                                'object': obj_ent,
                                'confidence': 0.8
                            }
                            
        except Exception as e:
            logger.debug(f"Error parsing REBEL output: {e}")
        
        return None
    
    def _match_entities_to_triplet(self, subj_text: str, obj_text: str, 
                                 ent1: Dict, ent2: Dict, context: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Match REBEL-extracted entities to our NER entities."""
        entities = [ent1, ent2]
        
        # Calculate similarity scores
        def text_similarity(text1: str, text2: str) -> float:
            text1_lower = text1.lower().strip()
            text2_lower = text2.lower().strip()
            
            # Exact match
            if text1_lower == text2_lower:
                return 1.0
            
            # Substring match
            if text1_lower in text2_lower or text2_lower in text1_lower:
                return 0.8
            
            # Word overlap
            words1 = set(text1_lower.split())
            words2 = set(text2_lower.split())
            if words1 & words2:
                return len(words1 & words2) / max(len(words1), len(words2))
            
            return 0.0
        
        # Find best matches
        best_subj = None
        best_obj = None
        best_subj_score = 0.5  # Minimum threshold
        best_obj_score = 0.5
        
        for ent in entities:
            subj_score = text_similarity(subj_text, ent['text'])
            obj_score = text_similarity(obj_text, ent['text'])
            
            if subj_score > best_subj_score:
                best_subj = ent
                best_subj_score = subj_score
            
            if obj_score > best_obj_score and ent != best_subj:
                best_obj = ent
                best_obj_score = obj_score
        
        return best_subj, best_obj
    
    def _extract_dependency_relation(self, text: str, ent1: Dict, ent2: Dict) -> Optional[Dict[str, Any]]:
        """Extract relations using dependency parsing."""
        try:
            # Use a simple English model for dependency parsing
            import spacy
            nlp_simple = spacy.load('en_core_web_sm')
            doc = nlp_simple(text)
            
            # Find tokens corresponding to entities
            ent1_tokens = self._find_entity_tokens(doc, ent1)
            ent2_tokens = self._find_entity_tokens(doc, ent2)
            
            if not ent1_tokens or not ent2_tokens:
                return None
            
            # Look for dependency paths between entities
            for token1 in ent1_tokens:
                for token2 in ent2_tokens:
                    relation = self._analyze_dependency_path(token1, token2)
                    if relation:
                        return relation
            
            return None
            
        except Exception as e:
            logger.debug(f"Dependency parsing failed: {e}")
            return None
    
    def _find_entity_tokens(self, doc, entity: Dict) -> List:
        """Find spaCy tokens corresponding to an entity."""
        tokens = []
        start_char = entity['start']
        end_char = entity['end']
        
        for token in doc:
            if token.idx >= start_char and token.idx + len(token.text) <= end_char:
                tokens.append(token)
        
        return tokens
    
    def _analyze_dependency_path(self, token1, token2) -> Optional[Dict[str, Any]]:
        """Analyze dependency path between two tokens."""
        # Common dependency patterns for biological relations
        patterns = {
            # Direct object relations
            'associated': {'predicate': 'associated_with', 'confidence': 0.8},
            'linked': {'predicate': 'associated_with', 'confidence': 0.8}, 
            'related': {'predicate': 'associated_with', 'confidence': 0.6},
            'involved': {'predicate': 'involved_in', 'confidence': 0.8},
            'regulates': {'predicate': 'regulates', 'confidence': 0.8},
            'activates': {'predicate': 'activates', 'confidence': 0.8},
            'inhibits': {'predicate': 'inhibits', 'confidence': 0.8},
            'causes': {'predicate': 'causes', 'confidence': 0.8},
            'mutated': {'predicate': 'mutated_in', 'confidence': 0.7},
            'expresses': {'predicate': 'expresses', 'confidence': 0.7},
        }
        
        # Find connecting verbs/relations
        path = []
        current = token1
        visited = set()
        
        # Traverse dependency tree to find connection
        while current and current != token2 and len(path) < 5:
            if current in visited:
                break
            visited.add(current)
            path.append(current)
            
            # Check if current token indicates a relation
            lemma = current.lemma_.lower()
            if lemma in patterns:
                return patterns[lemma]
            
            # Move to head or children
            if current.head != current:
                current = current.head
            else:
                break
        
        return None
    
    def _heuristic_relation_extraction(self, text: str, ent1: Dict, ent2: Dict) -> Optional[Dict[str, Any]]:
        """Enhanced heuristic-based relation extraction."""
        # Define entity type mappings (include CHEMICAL as potential gene)
        gene_labels = {'GENE', 'PROTEIN', 'CHEMICAL'}  # Many genes misclassified as CHEMICAL
        disease_labels = {'DISEASE', 'DISORDER'}
        chemical_labels = {'CHEMICAL', 'DRUG'}
        function_labels = {'FUNCTION', 'PROCESS', 'PATHWAY', 'CELLULAR_COMPONENT', 'MOLECULAR_FUNCTION'}
        
        ent1_label = ent1['label']
        ent2_label = ent2['label']
        
        # Get full text context
        context_lower = text.lower()
        
        # Pattern matching for relations
        relation_patterns = {
            'involved_in': ['involved in', 'plays a role in', 'participates in', 'functions in', 'role in'],
            'regulates': ['regulates', 'controls', 'modulates', 'affects', 'influences'],
            'associated_with': ['associated with', 'linked to', 'related to', 'implicated in', 'correlated with'],
            'causes': ['causes', 'leads to', 'results in', 'induces', 'triggers'],
            'inhibits': ['inhibits', 'suppresses', 'blocks', 'prevents', 'reduces'],
            'activates': ['activates', 'stimulates', 'enhances', 'promotes', 'increases'],
            'encodes': ['encodes', 'codes for', 'gene product', 'protein product'],
            'treats': ['treats', 'therapeutic', 'therapy', 'treatment', 'drug for'],
            'expresses': ['expresses', 'expression', 'expressed in', 'transcribed'],
            'mutated_in': ['mutations', 'mutated', 'variant', 'polymorphism'],
        }
        
        # Find the best matching predicate
        best_predicate = None
        best_confidence = 0.2
        
        # First try linguistic pattern matching
        linguistic_relation = self._extract_dependency_relation(text, ent1, ent2)
        if linguistic_relation:
            best_predicate = linguistic_relation['predicate']
            best_confidence = linguistic_relation['confidence']
        
        # Then try keyword patterns
        if not best_predicate or best_confidence < 0.6:
            for predicate, patterns in relation_patterns.items():
                for pattern in patterns:
                    if pattern in context_lower:
                        confidence = 0.7  # Higher confidence for pattern match
                        if confidence > best_confidence:
                            best_predicate = predicate
                            best_confidence = confidence
                            break
        
        # Default predicate if no pattern found but entities are compatible
        if not best_predicate:
            # Gene-Function/Process relations (very common in GeneRIF)
            if ((ent1_label in gene_labels and ent2_label in function_labels) or
                (ent1_label in function_labels and ent2_label in gene_labels)):
                best_predicate = 'involved_in'
                best_confidence = 0.4
            
            # Gene-Disease relations  
            elif ((ent1_label in gene_labels and ent2_label in disease_labels) or
                  (ent1_label in disease_labels and ent2_label in gene_labels)):
                best_predicate = 'associated_with'
                best_confidence = 0.4
                
            # Chemical-Disease relations
            elif ((ent1_label in chemical_labels and ent2_label in disease_labels) or
                  (ent1_label in disease_labels and ent2_label in chemical_labels)):
                best_predicate = 'associated_with'
                best_confidence = 0.3
            
            # Any other entity pairs - still extract with low confidence
            elif ent1_label != ent2_label:
                best_predicate = 'related_to'
                best_confidence = 0.25
        
        if best_predicate and best_confidence >= 0.2:
            # Determine subject/object based on entity types and order
            subject, object_ent = self._determine_subject_object(ent1, ent2, gene_labels)
            
            return {
                'subject': subject,
                'predicate': best_predicate,
                'object': object_ent,
                'confidence': best_confidence
            }
        
        return None
    
    def _determine_subject_object(self, ent1: Dict, ent2: Dict, gene_labels: Set[str]) -> Tuple[Dict, Dict]:
        """Determine which entity should be subject vs object."""
        # Prefer genes/proteins as subjects when possible
        if ent1['label'] in gene_labels and ent2['label'] not in gene_labels:
            return ent1, ent2
        elif ent2['label'] in gene_labels and ent1['label'] not in gene_labels:
            return ent2, ent1
        else:
            # Default to order in text (first entity as subject)
            if ent1['start'] < ent2['start']:
                return ent1, ent2
            else:
                return ent2, ent1
    
    def batch_extract_relations(self, texts: List[str], 
                              entities_list: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Extract relations from multiple texts."""
        relations_list = []
        
        for text, entities in tqdm(zip(texts, entities_list), 
                                  total=len(texts), desc="Extracting relations"):
            relations = self.extract_relations(text, entities)
            relations_list.append(relations)
        
        return relations_list
