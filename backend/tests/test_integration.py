import pytest
import os
from unittest.mock import patch, MagicMock
from main import ExternalMemorySystem

class TestIntegration:
    
    @pytest.fixture
    def system(self, tmp_path):
        # Create a temporary config
        config = {
            'chunking': {'chunk_size': 10, 'overlap': 2},
            'embedding': {'model_name': 'all-MiniLM-L6-v2', 'device': 'cpu'},
            'retrieval': {'top_k': 2, 'similarity_threshold': 0.0},
            'llm': {
                'provider': 'ollama',
                'ollama': {'model': 'llama2'}
            }
        }
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return ExternalMemorySystem(str(config_path))
    
    @patch('src.inference.LLMInference.generate')
    def test_full_pipeline(self, mock_generate, system):
        # Mock LLM response
        mock_generate.return_value = {
            'response': 'ML is about learning from data.',
            'tokens_used': {'total': 10},
            'latency_ms': 500
        }
        
        # 1. Ingest
        doc = "Machine learning is a subset of AI. It uses algorithms to find patterns. Deep learning is a part of ML."
        stats = system.ingest_document(doc, source="test_doc")
        assert stats['chunks_created'] > 0
        
        # 2. Query
        result = system.query("What is machine learning?")
        
        assert "learning from data" in result['answer']
        assert len(result['sources']) > 0
        assert result['sources'][0]['metadata']['source'] == "test_doc"
    
    def test_save_load_integration(self, system, tmp_path):
        doc = "Some persistent data."
        system.ingest_document(doc, source="source1")
        
        save_path = str(tmp_path / "memory")
        system.save_memory(save_path)
        
        # New system
        new_system = ExternalMemorySystem("config.yaml") # USes base config
        new_system.load_memory(save_path)
        
        assert new_system.get_statistics()['total_chunks'] == system.get_statistics()['total_chunks']
