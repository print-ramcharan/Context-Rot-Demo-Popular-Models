import sys
from pathlib import Path
from main import ExternalMemorySystem

def calculate_grounding_score(system, question: str, answer: str, context: str) -> float:
    """
    Use the LLM as a judge to score how grounded the answer is in the context.
    Returns a score from 0.0 to 1.0.
    """
    if not context:
        return 0.0
        
    judge_prompt = f"""
    You are an objective evaluator. Rate the 'Groundedness' of the following answer based ONLY on the provided context.
    
    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: {answer}
    
    CRITERIA:
    1. Does the answer contain information NOT found in the context? (If yes, penalize)
    2. Does the answer contradict the context? (If yes, 0 score)
    3. Is the answer fully supported by the context?
    
    Output ONLY a JSON object: {{"score": float, "reason": string}}
    Lower score if the answer uses general knowledge instead of the specific context provided.
    Score is 1.0 if perfectly grounded, 0.0 if entirely hallucinated or unsupported.
    """
    
    try:
        response = system.llm.generate(judge_prompt)
        import json
        import re
        # Clean up possible markdown in response
        clean_text = re.sub(r'```json\n?|\n?```', '', response['response'])
        data = json.loads(clean_text)
        return float(data.get('score', 0.0)), data.get('reason', '')
    except Exception as e:
        return 0.0, f"Error calculating score: {e}"

def compare_query(question: str):
    """
    Compare LLM output for the same question with and without context.
    """
    print(f"\nQuestion: {question}")
    print("=" * 60)
    
    # Initialize the system
    system = ExternalMemorySystem()
    
    # 0. Retrieve Context First (for evaluation)
    retrieved_context = system.query(question, k=3)
    context_text = "\n".join([c['text'] for c in retrieved_context['sources']])
    
    # 1. Without Context (Zero-shot)
    print("\n[1] WITHOUT CONTEXT (Zero-shot)")
    zero_shot_prompt = f"Question: {question}\nAnswer based on your general knowledge:"
    res_no_context = system.llm.generate(zero_shot_prompt)
    ans_no_context = res_no_context['response']
    score_no, reason_no = calculate_grounding_score(system, question, ans_no_context, context_text)
    
    print("\nAnswer:")
    print(ans_no_context[:300] + "...")
    print(f"\n>> GROUNDING SCORE: {score_no * 100:.1f}/100")
    print(f">> REASON: {reason_no}")
    print("-" * 40)
    
    # 2. With Context (RAG)
    print("\n[2] WITH CONTEXT (RAG)")
    ans_with_context = retrieved_context['answer']
    score_with, reason_with = calculate_grounding_score(system, question, ans_with_context, context_text)
    
    print("\nAnswer:")
    print(ans_with_context)
    print(f"\n>> GROUNDING SCORE: {score_with * 100:.1f}/100")
    print(f">> REASON: {reason_with}")
    print("-" * 40)
    
    print(f"\nSUMMARY: Context improved accuracy by {(score_with - score_no) * 100:.1f}%")

if __name__ == "__main__":
    test_question = "What is the main contribution of HippoRAG?"
    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])
    
    compare_query(test_question)
