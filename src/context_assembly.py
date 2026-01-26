class ContextAssembler:
    """
    Assembles retrieved chunks into formatted prompts for LLMs.
    """
    
    def __init__(self, max_context_length: int = 4000):
        """
        Initialize assembler with token constraints.
        
        Args:
            max_context_length (int): Maximum characters for context
        """
        self.max_context_length = max_context_length
        self.templates = {
            "default": (
                "You are a helpful assistant. Use the following context to answer the question.\n\n"
                "CONTEXT:\n"
                "{context}\n\n"
                "QUESTION:\n"
                "{query}\n\n"
                "Provide a concise and accurate answer based on the context above."
            ),
            "instructional": (
                "Answer the user's question by following these steps:\n"
                "1. Read the provided context carefully.\n"
                "2. Identify the key facts.\n"
                "3. Formulate a structured response.\n\n"
                "CONTEXT:\n"
                "{context}\n\n"
                "QUESTION:\n"
                "{query}"
            )
        }
    
    def assemble_prompt(self, query: str, retrieved_chunks: list[dict],
                       template_name: str = "default") -> str:
        """
        Create formatted prompt from query and retrieved chunks.
        
        Args:
            query (str): User question/query
            retrieved_chunks (list[dict]): Chunks from retrieval
            template_name (str): Prompt template name
            
        Returns:
            str: Formatted prompt ready for LLM
        """
        # 1. Truncate chunks if they exceed max length
        valid_chunks = self.truncate_to_fit(retrieved_chunks, self.max_context_length)
        
        # 2. Format context string
        context_parts = []
        for i, chunk in enumerate(valid_chunks):
            meta = chunk.get('metadata', {})
            source = meta.get('source', f"Document {i+1}")
            ext = meta.get('extension', 'txt')
            score = chunk.get('score', 0)
            text = chunk.get('text', '')
            
            header = f"[Source: {source} (.{ext}) - Score: {score:.3f}]"
            context_parts.append(f"{header}\n{text}")
            
        context_str = "\n\n".join(context_parts)
        
        # 3. Apply template
        template = self.templates.get(template_name, self.templates["default"])
        return template.format(context=context_str, query=query)
    
    def create_conversational_prompt(self, query: str, 
                                     retrieved_chunks: list[dict],
                                     conversation_history: list[dict] = None) -> str:
        """
        Create prompt that includes conversation history.
        """
        context_prompt = self.assemble_prompt(query, retrieved_chunks)
        
        if not conversation_history:
            return context_prompt
            
        history_str = ""
        for msg in conversation_history:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
            
        full_prompt = (
            f"CONVERSATION HISTORY:\n{history_str}\n"
            f"CURRENT TASK:\n{context_prompt}"
        )
        return full_prompt
    
    def truncate_to_fit(self, chunks: list[dict], 
                       max_length: int) -> list[dict]:
        """
        Truncate or select chunks to fit within length constraint.
        Prioritizes higher-scored chunks.
        """
        current_length = 0
        valid_chunks = []
        
        # Sort by score DESC just in case they aren't already
        # (Usually SemanticRetriever does this)
        sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        for chunk in sorted_chunks:
            chunk_len = len(chunk['text'])
            if current_length + chunk_len > max_length:
                if current_length == 0:
                    # If first chunk is already too long, take a prefix
                    valid_chunks.append({
                        **chunk,
                        'text': chunk['text'][:max_length]
                    })
                break
            valid_chunks.append(chunk)
            current_length += chunk_len
            
        return valid_chunks
    
    def add_citations(self, response: str, 
                     chunks: list[dict]) -> dict:
        """
        Add source citations to LLM response.
        """
        citations = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get('metadata', {})
            source = meta.get('source', f"Chunk {i+1}")
            citations.append(f"[{i+1}] {source}")
            
        return {
            'response': response,
            'citations': citations
        }
