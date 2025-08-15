"""
Filtering and post-processing of extracted triplets.
"""

import logging
from typing import List, Dict, Set, Any
import pandas as pd

logger = logging.getLogger(__name__)


class TripleFilter:
    """Filter and post-process extracted relation triplets."""
    
    def __init__(self, 
                 min_confidence: float = 0.5,
                 filter_gene_gene: bool = True,
                 filter_gene_as_object: bool = True):
        """
        Initialize triplet filter.
        
        Args:
            min_confidence: Minimum confidence threshold for relations
            filter_gene_gene: Whether to filter out gene-gene relations
            filter_gene_as_object: Whether to filter out relations where gene is object
        """
        self.min_confidence = min_confidence
        self.filter_gene_gene = filter_gene_gene
        self.filter_gene_as_object = filter_gene_as_object
        
        # Define entity type mappings
        self.gene_labels = {'GENE', 'PROTEIN'}
        self.disease_labels = {'DISEASE', 'DISORDER'}
        self.chemical_labels = {'CHEMICAL', 'DRUG'}
        self.function_labels = {'FUNCTION', 'PROCESS', 'PATHWAY'}
        
        # Blacklisted predicates (too general or noisy)
        self.blacklisted_predicates = {
            'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'related_to', 'mentioned', 'occurs', 'exists'
        }
        
        # Preferred predicates for specific entity pairs
        self.predicate_preferences = {
            ('GENE', 'DISEASE'): ['associated_with', 'causes', 'linked_to', 'implicated_in'],
            ('GENE', 'FUNCTION'): ['involved_in', 'regulates', 'encodes', 'participates_in'],
            ('CHEMICAL', 'DISEASE'): ['treats', 'inhibits', 'therapeutic_for'],
            ('PROTEIN', 'PATHWAY'): ['participates_in', 'regulates', 'activates', 'inhibits']
        }
    
    def filter_triplets(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply all filters to a list of relation triplets.
        
        Args:
            relations: List of relation dictionaries
            
        Returns:
            Filtered list of relations
        """
        if not relations:
            return []
        
        filtered = relations.copy()
        
        # Apply confidence filter
        filtered = self._filter_by_confidence(filtered)
        
        # Apply gene-gene filter
        if self.filter_gene_gene:
            filtered = self._filter_gene_gene_relations(filtered)
        
        # Apply gene-as-object filter  
        if self.filter_gene_as_object:
            filtered = self._filter_gene_as_object(filtered)
        
        # Filter blacklisted predicates
        filtered = self._filter_blacklisted_predicates(filtered)
        
        # Deduplicate relations
        filtered = self._deduplicate_relations(filtered)
        
        logger.debug(f"Filtered {len(relations)} -> {len(filtered)} relations")
        return filtered
    
    def _filter_by_confidence(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter relations by confidence threshold."""
        return [r for r in relations if r.get('confidence', 0) >= self.min_confidence]
    
    def _filter_gene_gene_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out gene-gene relations."""
        filtered = []
        for rel in relations:
            subj_label = rel['subject']['label']
            obj_label = rel['object']['label']
            
            # Skip if both subject and object are genes/proteins
            if subj_label in self.gene_labels and obj_label in self.gene_labels:
                continue
            
            filtered.append(rel)
        
        return filtered
    
    def _filter_gene_as_object(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out relations where gene is the object."""
        filtered = []
        for rel in relations:
            obj_label = rel['object']['label']
            
            # Skip if object is a gene/protein (we want genes as subjects)
            if obj_label in self.gene_labels:
                continue
            
            filtered.append(rel)
        
        return filtered
    
    def _filter_blacklisted_predicates(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out relations with blacklisted predicates."""
        filtered = []
        for rel in relations:
            predicate = rel['predicate'].lower()
            
            if predicate not in self.blacklisted_predicates:
                filtered.append(rel)
        
        return filtered
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relations."""
        seen = set()
        filtered = []
        
        for rel in relations:
            # Create a unique key for the relation
            subj_text = rel['subject']['text'].lower()
            pred_text = rel['predicate'].lower()
            obj_text = rel['object']['text'].lower()
            
            key = (subj_text, pred_text, obj_text)
            
            if key not in seen:
                seen.add(key)
                filtered.append(rel)
        
        return filtered
    
    def normalize_predicates(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize and standardize predicates."""
        predicate_mappings = {
            # Causation
            'causes': 'causes',
            'leads_to': 'causes',
            'results_in': 'causes',
            'induces': 'causes',
            
            # Association
            'associated_with': 'associated_with',
            'linked_to': 'associated_with',
            'related_to': 'associated_with',
            'implicated_in': 'associated_with',
            
            # Regulation
            'regulates': 'regulates',
            'controls': 'regulates',
            'modulates': 'regulates',
            
            # Inhibition
            'inhibits': 'inhibits',
            'blocks': 'inhibits',
            'suppresses': 'inhibits',
            'downregulates': 'inhibits',
            
            # Activation
            'activates': 'activates',
            'upregulates': 'activates',
            'enhances': 'activates',
            'stimulates': 'activates',
            
            # Treatment
            'treats': 'treats',
            'therapeutic_for': 'treats',
            'therapy_for': 'treats',
            
            # Function
            'involved_in': 'involved_in',
            'participates_in': 'involved_in',
            'part_of': 'involved_in',
        }
        
        normalized = []
        for rel in relations:
            predicate = rel['predicate'].lower()
            normalized_pred = predicate_mappings.get(predicate, predicate)
            
            rel_copy = rel.copy()
            rel_copy['predicate'] = normalized_pred
            normalized.append(rel_copy)
        
        return normalized
    
    def batch_filter_triplets(self, relations_list: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Apply filters to multiple lists of relations."""
        return [self.filter_triplets(relations) for relations in relations_list]
    
    def get_filter_stats(self, original_relations: List[Dict[str, Any]], 
                        filtered_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about filtering results."""
        original_count = len(original_relations)
        filtered_count = len(filtered_relations)
        
        # Count by predicate
        original_predicates = {}
        filtered_predicates = {}
        
        for rel in original_relations:
            pred = rel['predicate']
            original_predicates[pred] = original_predicates.get(pred, 0) + 1
        
        for rel in filtered_relations:
            pred = rel['predicate']
            filtered_predicates[pred] = filtered_predicates.get(pred, 0) + 1
        
        return {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'retention_rate': filtered_count / original_count if original_count > 0 else 0,
            'original_predicates': original_predicates,
            'filtered_predicates': filtered_predicates
        }


class QualityFilter:
    """Additional quality filters for triplets."""
    
    def __init__(self, 
                 min_entity_length: int = 2,
                 max_entity_length: int = 100,
                 min_predicate_length: int = 2):
        """
        Initialize quality filter.
        
        Args:
            min_entity_length: Minimum entity text length
            max_entity_length: Maximum entity text length
            min_predicate_length: Minimum predicate length
        """
        self.min_entity_length = min_entity_length
        self.max_entity_length = max_entity_length
        self.min_predicate_length = min_predicate_length
        
        # Common noise patterns to filter out
        self.noise_patterns = {
            'entities': ['figure', 'table', 'et al', 'study', 'data', 'result'],
            'predicates': ['a', 'an', 'the', 'of', 'in', 'at', 'to', 'for']
        }
    
    def apply_quality_filters(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply quality filters to relations."""
        filtered = []
        
        for rel in relations:
            if self._is_high_quality_relation(rel):
                filtered.append(rel)
        
        return filtered
    
    def _is_high_quality_relation(self, relation: Dict[str, Any]) -> bool:
        """Check if a relation meets quality criteria."""
        subj_text = relation['subject']['text'].strip()
        pred_text = relation['predicate'].strip()
        obj_text = relation['object']['text'].strip()
        
        # Check entity length constraints
        if (len(subj_text) < self.min_entity_length or 
            len(subj_text) > self.max_entity_length or
            len(obj_text) < self.min_entity_length or
            len(obj_text) > self.max_entity_length):
            return False
        
        # Check predicate length
        if len(pred_text) < self.min_predicate_length:
            return False
        
        # Check for noise patterns
        subj_lower = subj_text.lower()
        obj_lower = obj_text.lower()
        pred_lower = pred_text.lower()
        
        for noise in self.noise_patterns['entities']:
            if noise in subj_lower or noise in obj_lower:
                return False
        
        for noise in self.noise_patterns['predicates']:
            if pred_lower == noise:
                return False
        
        # Check for numeric-only entities (usually not meaningful)
        if subj_text.isdigit() or obj_text.isdigit():
            return False
        
        return True
