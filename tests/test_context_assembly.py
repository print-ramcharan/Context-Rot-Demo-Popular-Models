import pytest
from src.context_assembly import ContextAssembler

class TestContextAssembler:
    
    def test_assemble_prompt(self):
        assembler = ContextAssembler(max_context_length=1000)
        chunks = [
            {'text': 'Paris is the capital of France.', 'score': 0.95},
            {'text': 'The Eiffel Tower is in Paris.', 'score': 0.85}
        ]
        query = "Where is the Eiffel Tower?"
        prompt = assembler.assemble_prompt(query, chunks)
        
        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt
        assert chunks[0]['text'] in prompt
        assert chunks[1]['text'] in prompt
        assert query in prompt
        assert "[Chunk 1 - Score: 0.950]" in prompt
    
    def test_truncation(self):
        # Max length only allows one chunk
        assembler = ContextAssembler(max_context_length=50)
        chunks = [
            {'text': 'This is a long chunk that should fit.', 'score': 0.9},
            {'text': 'This is another chunk that should be truncated.', 'score': 0.8}
        ]
        # Text 1 is ~37 chars. Text 2 would push it over 50.
        valid = assembler.truncate_to_fit(chunks, 50)
        assert len(valid) == 1
        assert valid[0]['text'] == chunks[0]['text']
    
    def test_conversational_prompt(self):
        assembler = ContextAssembler()
        history = [
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello! How can I help?'}
        ]
        chunks = [{'text': 'Context data', 'score': 1.0}]
        prompt = assembler.create_conversational_prompt("Tell me more.", chunks, history)
        
        assert "CONVERSATION HISTORY:" in prompt
        assert "USER: Hi" in prompt
        assert "ASSISTANT: Hello!" in prompt
        assert "Context data" in prompt
    
    def test_citations(self):
        assembler = ContextAssembler()
        chunks = [
            {'text': 'text1', 'metadata': {'source': 'doc1.txt'}},
            {'text': 'text2', 'metadata': {'source': 'doc2.txt'}}
        ]
        result = assembler.add_citations("The answer is X.", chunks)
        assert len(result['citations']) == 2
        assert "[1] doc1.txt" in result['citations']
        assert "[2] doc2.txt" in result['citations']
