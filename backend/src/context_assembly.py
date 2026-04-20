import re

# Pre-compiled patterns for performance
_WORD_PATTERN = re.compile(r"\b\w+\b")
_SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

class ContextAssembler:
    """
    Assembles retrieved chunks into formatted prompts for LLMs.
    """
    # Minimal abbreviation list; extend for domain-specific content as needed.
    ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "e.g.", "i.e."]
    STOP_WORDS = {
        "is", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", 
        "of", "by", "from", "as", "it", "its", "they", "them", "their", "this", "that"
    }
    
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
        """
        Extract the most query-relevant sentences from a chunk.
        Short chunks are returned intact to avoid shearing off critical context.
        When filtering, adjacent sentences are included for continuity.
        """
        # Short chunks: return as-is to avoid destroying context
        if len(chunk_text) <= 500:
            return chunk_text

        sentences = self._split_sentences(chunk_text)
        if not sentences or len(sentences) <= max_sentences:
            return chunk_text

        query_terms = set(_WORD_PATTERN.findall(query.lower()))
        # Filter out common stop words to focus on content-bearing terms
        relevant_query_terms = {term for term in query_terms if term not in self.STOP_WORDS}
        
        # Verb root mapping for common RAG queries
        VERB_ROOTS = {"find": "found", "finds": "found", "found": "find", "get": "got", "gets": "got"}
        enriched_terms = set(relevant_query_terms)
        for term in relevant_query_terms:
            if term in VERB_ROOTS:
                enriched_terms.add(VERB_ROOTS[term])
        
        if not enriched_terms:
            enriched_terms = query_terms
            
        if not enriched_terms:
            return " ".join(sentences[:max_sentences])

        scored = []
        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            terms = set(_WORD_PATTERN.findall(sentence_lower))
            
            # Exact match score
            exact_matches = len(terms.intersection(enriched_terms))
            
            # Prefix match for robustness
            prefix_matches = 0
            for q_term in enriched_terms:
                if len(q_term) < 4: continue
                if any(q_term[:4] in s_term for s_term in terms if len(s_term) > 3):
                    prefix_matches += 0.5
            
            score = exact_matches + prefix_matches
            scored.append((idx, score, sentence))

        # If no query term matches at all, return the first N sentences
        if all(score == 0 for _, score, _ in scored):
            return " ".join(sentences[:max_sentences])

        # Sort by relevance score descending, then by original position
        scored.sort(key=lambda x: (-x[1], x[0]))

        # Collect top sentence indices AND their neighbors for context continuity
        selected_indices = set()
        for idx, score, _ in scored:
            if len(selected_indices) >= max_sentences:
                break
            if score > 0:
                selected_indices.add(idx)
                # Include MORE adjacent sentences to preserve surrounding context
                # RAG needs the "story" flow to be accurate.
                for offset in [-1, 1, 2]:
                    neighbor = idx + offset
                    if 0 <= neighbor < len(sentences):
                        selected_indices.add(neighbor)

        # Sort by original position and join
        ordered = sorted(selected_indices)[:max_sentences + 4]  # Allow larger overflow for neighbors
        return " ".join([sentences[i] for i in ordered if i < len(sentences)])

    def _split_sentences(self, text: str) -> list[str]:
        placeholder = "<<CONTEXT_ROT_DOT>>"
        safe_text = text
        for abbr in self.ABBREVIATIONS:
            safe_text = safe_text.replace(abbr, abbr.replace(".", placeholder))
        sentences = _SENTENCE_SPLIT_PATTERN.split(safe_text.strip())
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

        for i, chunk in enumerate(sorted_chunks):
            if len(compressed) >= max_chunks_after_compaction:
                break
            
            # If a chunk is highly relevant (rank 0 or 1), or has a very high score,
            # we keep more of it to ensure precision.
            score = chunk.get("score", 0)
            rank = chunk.get("rank", i)
            
            if rank < 2 or score > 0.04:  # RRF scores > 0.04 are typically very strong
                current_max_sentences = max_sentences + 4
            else:
                current_max_sentences = max_sentences

            compact_text = self.extract_relevant_sentences(query, chunk.get("text", ""), max_sentences=current_max_sentences)
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
