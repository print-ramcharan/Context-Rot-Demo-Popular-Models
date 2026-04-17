import re

class ContextAssembler:
    """
    Assembles retrieved chunks into formatted prompts for LLMs.
    """
    ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "e.g.", "i.e."]
    
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
                       template_name: str = "default",
                       compact: bool = False,
                       compact_config: dict | None = None) -> str:
        """
        Create formatted prompt from query and retrieved chunks.
        
        Args:
            query (str): User question/query
            retrieved_chunks (list[dict]): Chunks from retrieval
            template_name (str): Prompt template name
            
        Returns:
            str: Formatted prompt ready for LLM
        """
        # 1. Optionally compress context
        if compact:
            cfg = compact_config or {}
            retrieved_chunks = self.compress_context(
                query,
                retrieved_chunks,
                max_context_chars=cfg.get("max_context_chars", 2600),
                max_sentences=cfg.get("max_sentences", 2),
                max_chunks_after_compaction=cfg.get("max_chunks", 4)
            )

        # 2. Truncate chunks if they exceed max length
        valid_chunks = self.truncate_to_fit(retrieved_chunks, self.max_context_length)
        
        # 3. Format context string
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
        
        # 4. Apply template
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

    def extract_relevant_sentences(self, query: str, chunk_text: str, max_sentences: int = 2) -> str:
        sentences = self._split_sentences(chunk_text)
        if not sentences:
            return chunk_text

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        if not query_terms:
            return " ".join(sentences[:max_sentences])

        scored = []
        for idx, sentence in enumerate(sentences):
            terms = set(re.findall(r"\b\w+\b", sentence.lower()))
            score = len(terms.intersection(query_terms))
            scored.append((idx, score, sentence))

        if all(score == 0 for _, score, _ in scored):
            return " ".join(sentences[:max_sentences])
        scored.sort(key=lambda x: (-x[1], x[0]))
        top = sorted(scored[:max_sentences], key=lambda x: x[0])
        return " ".join([t[2] for t in top])

    def _split_sentences(self, text: str) -> list[str]:
        placeholder = "<DOT>"
        safe_text = text
        for abbr in self.ABBREVIATIONS:
            safe_text = safe_text.replace(abbr, abbr.replace(".", placeholder))
        sentences = re.split(r'(?<=[.!?])\s+', safe_text.strip())
        return [s.replace(placeholder, ".").strip() for s in sentences if s.strip()]

    def compress_context(self, query: str, chunks: list[dict],
                         max_context_chars: int = 2600,
                         max_sentences: int = 2,
                         max_chunks_after_compaction: int = 4) -> list[dict]:
        if not chunks:
            return []

        sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
        compressed = []
        current_len = 0

        for chunk in sorted_chunks:
            if len(compressed) >= max_chunks_after_compaction:
                break
            compact_text = self.extract_relevant_sentences(query, chunk.get("text", ""), max_sentences=max_sentences)
            if not compact_text:
                continue
            if current_len + len(compact_text) > max_context_chars:
                break
            updated = dict(chunk)
            updated["text"] = compact_text
            compressed.append(updated)
            current_len += len(compact_text)

        return compressed
    
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
