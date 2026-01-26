from typing import List, Dict, Tuple
import time
import json
import numpy as np
from main import ExternalMemorySystem

class EvaluationFramework:
    """
    Comprehensive evaluation for external memory system.
    """
    
    def __init__(self, system: ExternalMemorySystem):
        self.system = system
        self.results = {
            'retrieval_metrics': {},
            'answer_quality': {},
            'efficiency': {}
        }
    
    def evaluate_retrieval_precision(self, 
                                    queries: list[str],
                                    ground_truth_docs: list[str]) -> dict:
        """
        Measure if the retrieved chunks contain the ground truth source.
        """
        hits = 0
        total = len(queries)
        
        for query, expected_source in zip(queries, ground_truth_docs):
            results = self.system.retriever.retrieve(query, k=3)
            # Check if any retrieved chunk came from the expected source
            for res in results:
                if res['metadata'].get('source') == expected_source:
                    hits += 1
                    break
        
        precision_at_k = hits / total if total > 0 else 0
        self.results['retrieval_metrics']['precision_at_3'] = precision_at_k
        return self.results['retrieval_metrics']
    
    def benchmark_latency(self, num_queries: int = 5) -> dict:
        """
        Measure system latency components.
        """
        total_retrieval_time = 0
        total_llm_time = 0
        
        test_queries = ["test query"] * num_queries
        
        for q in test_queries:
            start = time.time()
            retrieved = self.system.retriever.retrieve(q, k=3)
            total_retrieval_time += (time.time() - start) * 1000
            
            # For LLM, we use the recorded latency in the response if available
            res = self.system.query(q, k=1)
            total_llm_time += res.get('latency_ms', 0)
            
        avg_retrieval = total_retrieval_time / num_queries
        avg_llm = total_llm_time / num_queries
        
        self.results['efficiency'] = {
            'avg_retrieval_ms': avg_retrieval,
            'avg_llm_ms': avg_llm,
            'total_latency_ms': avg_retrieval + avg_llm
        }
        return self.results['efficiency']
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """
        Save evaluation results to file.
        """
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Evaluation report saved to {output_path}")

def run_evaluation():
    # Setup system with mock/small models for testing the framework itself
    system = ExternalMemorySystem()
    evaluator = EvaluationFramework(system)
    
    # Ingest some dummy docs
    system.ingest_document("The quick brown fox jumps over the lazy dog.", source="fox.txt")
    system.ingest_document("To be or not to be, that is the question.", source="hamlet.txt")
    
    # Run metrics
    print("Evaluating retrieval...")
    evaluator.evaluate_retrieval_precision(
        queries=["Who jumps over the dog?", "What is the question?"],
        ground_truth_docs=["fox.txt", "hamlet.txt"]
    )
    
    print("Benchmarking latency...")
    evaluator.benchmark_latency()
    
    evaluator.generate_report()

if __name__ == "__main__":
    run_evaluation()
