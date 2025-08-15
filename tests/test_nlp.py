"""Test NLP pipeline functionality."""

import pytest
from unittest.mock import Mock, patch
from gene_rif_graph.nlp import BioNERExtractor, RelationExtractor


class TestBioNERExtractor:
    """Test biomedical NER functionality."""
    
    @patch('gene_rif_graph.nlp.spacy.load')
    def test_init(self, mock_spacy_load):
        """Test NER extractor initialization."""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        extractor = BioNERExtractor(model_name="test_model")
        assert extractor.model_name == "test_model"
        assert extractor.nlp == mock_nlp
    
    @patch('gene_rif_graph.nlp.spacy.load')
    def test_extract_entities(self, mock_spacy_load):
        """Test entity extraction."""
        # Mock spaCy model and entities
        mock_ent = Mock()
        mock_ent.text = "TP53"
        mock_ent.label_ = "GENE"
        mock_ent.start_char = 0
        mock_ent.end_char = 4
        
        mock_doc = Mock()
        mock_doc.ents = [mock_ent]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_nlp.pipe_names = []
        mock_spacy_load.return_value = mock_nlp
        
        extractor = BioNERExtractor()
        entities = extractor.extract_entities("TP53 gene")
        
        assert len(entities) == 1
        assert entities[0]['text'] == "TP53"
        assert entities[0]['label'] == "GENE"
        assert entities[0]['start'] == 0
        assert entities[0]['end'] == 4
    
    @patch('gene_rif_graph.nlp.spacy.load')
    def test_batch_extract_entities(self, mock_spacy_load):
        """Test batch entity extraction."""
        # Mock spaCy pipeline
        mock_ent1 = Mock()
        mock_ent1.text = "TP53"
        mock_ent1.label_ = "GENE"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 4
        
        mock_ent2 = Mock()
        mock_ent2.text = "cancer"
        mock_ent2.label_ = "DISEASE"
        mock_ent2.start_char = 0
        mock_ent2.end_char = 6
        
        mock_doc1 = Mock()
        mock_doc1.ents = [mock_ent1]
        
        mock_doc2 = Mock()
        mock_doc2.ents = [mock_ent2]
        
        mock_nlp = Mock()
        mock_nlp.pipe.return_value = [mock_doc1, mock_doc2]
        mock_nlp.pipe_names = []
        mock_spacy_load.return_value = mock_nlp
        
        extractor = BioNERExtractor()
        results = extractor.batch_extract_entities(["TP53 gene", "cancer disease"])
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert results[0][0]['text'] == "TP53"
        assert results[1][0]['text'] == "cancer"


class TestRelationExtractor:
    """Test relation extraction functionality."""
    
    @patch('gene_rif_graph.nlp.AutoTokenizer')
    @patch('gene_rif_graph.nlp.AutoModelForSequenceClassification')
    def test_init_biobert(self, mock_model, mock_tokenizer):
        """Test initialization with BioBERT model."""
        extractor = RelationExtractor(model_name="dmis-lab/biobert-base-cased-v1.1")
        assert "biobert" in extractor.model_name.lower()
        assert extractor.device in ['cpu', 'cuda']
    
    @patch('gene_rif_graph.nlp.pipeline')
    def test_init_rebel(self, mock_pipeline):
        """Test initialization with REBEL model."""
        extractor = RelationExtractor(model_name="rebel-large")
        assert "rebel" in extractor.model_name.lower()
    
    def test_extract_relations(self):
        """Test relation extraction between entities."""
        # Mock the relation extractor
        extractor = RelationExtractor(model_name="test-model")
        extractor._extract_relation_pair = Mock(return_value={
            'subject': {'text': 'TP53', 'label': 'GENE'},
            'predicate': 'associated_with',
            'object': {'text': 'cancer', 'label': 'DISEASE'},
            'confidence': 0.8
        })
        
        entities = [
            {'text': 'TP53', 'label': 'GENE', 'start': 0, 'end': 4},
            {'text': 'cancer', 'label': 'DISEASE', 'start': 20, 'end': 26}
        ]
        
        relations = extractor.extract_relations("TP53 is linked to cancer", entities)
        
        assert len(relations) == 1
        assert relations[0]['subject']['text'] == 'TP53'
        assert relations[0]['object']['text'] == 'cancer'
        assert relations[0]['predicate'] == 'associated_with'
    
    def test_heuristic_relation_extraction(self):
        """Test heuristic-based relation extraction."""
        extractor = RelationExtractor(model_name="test-model")
        
        # Test gene-disease relation
        text = "TP53 is associated with cancer development"
        ent1 = {'text': 'TP53', 'label': 'GENE', 'start': 0, 'end': 4}
        ent2 = {'text': 'cancer', 'label': 'DISEASE', 'start': 20, 'end': 26}
        
        relation = extractor._heuristic_relation_extraction(text, ent1, ent2)
        
        assert relation is not None
        assert relation['subject']['text'] == 'TP53'
        assert relation['object']['text'] == 'cancer'
        assert relation['predicate'] == 'associated_with'
        assert relation['confidence'] > 0
    
    def test_mark_entities_in_text(self):
        """Test entity marking in text."""
        extractor = RelationExtractor(model_name="test-model")
        
        text = "TP53 gene causes cancer"
        ent1 = {'text': 'TP53', 'label': 'GENE', 'start': 0, 'end': 4}
        ent2 = {'text': 'cancer', 'label': 'DISEASE', 'start': 17, 'end': 23}
        
        marked_text = extractor._mark_entities_in_text(text, ent1, ent2, 0)
        
        assert '<ent>' in marked_text
        assert '</ent>' in marked_text
        # Should have markers around both entities
