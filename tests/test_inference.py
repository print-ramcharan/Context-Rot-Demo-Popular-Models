import pytest
from unittest.mock import Mock, patch
from src.inference import LLMInference

class TestLLMInference:
    
    @patch('ollama.Client')
    def test_ollama_inference(self, mock_ollama):
        # Setup mock
        mock_client = mock_ollama.return_value
        mock_client.generate.return_value = {
            'response': 'Blue',
            'prompt_eval_count': 5,
            'eval_count': 1
        }
        
        llm = LLMInference(provider="ollama", model="llama2")
        result = llm.generate("What color is the sky?")
        
        assert result['response'] == 'Blue'
        assert result['provider'] == 'ollama'
        assert result['tokens_used']['total'] == 6
    
    @patch('anthropic.Anthropic')
    def test_anthropic_inference(self, mock_anthropic):
        # Setup mock
        mock_client = mock_anthropic.return_value
        mock_response = Mock()
        mock_response.content = [Mock(text="Paris")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response
        
        llm = LLMInference(provider="anthropic", model="claude-3-sonnet", config={'api_key': 'test'})
        result = llm.generate("Capital of France?")
        
        assert result['response'] == 'Paris'
        assert result['tokens_used']['total'] == 15
    
    @patch('openai.OpenAI')
    def test_openai_inference(self, mock_openai):
        # Setup mock
        mock_client = mock_openai.return_value
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="4"))]
        mock_response.usage = Mock(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = LLMInference(provider="openai", model="gpt-4", config={'api_key': 'test'})
        result = llm.generate("2+2")
        
        assert result['response'] == "4"
        assert result['tokens_used']['total'] == 3

    def test_unsupported_provider(self):
        with pytest.raises(ValueError):
            LLMInference(provider="unknown")
