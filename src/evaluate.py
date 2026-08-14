"""
Main evaluation runner for the Fair Work RAG system.
Processes documents, builds index, generates evaluation set, and computes metrics.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import FairWorkRAGPipeline
from embedding import EmbeddingModel, EmbeddingManager
from retrieval import HybridRetriever
from evaluation.eval_generator import EvaluationSetGenerator
from evaluation.eval_metrics import EvaluationMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs the complete evaluation pipeline."""
    
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "data/processed",
        eval_dir: str = "evaluation_set",
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            data_dir: Directory containing raw PDFs
            output_dir: Directory for processed outputs
            eval_dir: Directory for evaluation sets
            embedding_model: OpenAI embedding model name
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.eval_dir = Path(eval_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pipeline = FairWorkRAGPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            embedding_model=embedding_model
        )
        
        self.generator = EvaluationSetGenerator(output_dir=str(self.eval_dir))
        
        logger.info("Initialized EvaluationRunner")
    
    def run_evaluation(
        self,
        pdf_paths: List[str],
        num_queries: int = 100,
        k_values: List[int] = None
    ) -> Dict:
        """
        Run the complete evaluation pipeline.
        
        Args:
            pdf_paths: List of PDF file paths
            num_queries: Number of queries to generate
            k_values: K values for metrics computation
            
        Returns:
            Dict of evaluation results
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Process documents
        logger.info("\n[Step 1] Processing documents...")
        chunks = self.pipeline.process_pdfs(pdf_paths)
        logger.info(f"Processed {len(chunks)} chunks")
        
        # Step 2: Build index
        logger.info("\n[Step 2] Building retrieval index...")
        self.pipeline.build_index(chunks)
        
        # Save chunks for evaluation
        chunks_file = self.output_dir / "chunks.json"
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved chunks to {chunks_file}")
        
        # Step 3: Generate evaluation set
        logger.info("\n[Step 3] Generating evaluation set...")
        self.generator.generate_queries(chunks, num_queries=num_queries)
        self.generator.generate_annotations(chunks, auto_annotate=True)
        self.generator.save_evaluation_set()
        
        queries_file = self.eval_dir / "queries.json"
        annotations_file = self.eval_dir / "annotations.json"
        
        queries, annotations = self.generator.queries, self.generator.annotations
        logger.info(f"Generated {len(queries)} queries and {len(annotations)} annotations")
        
        # Step 4: Run retrieval and compute metrics
        logger.info("\n[Step 4] Computing retrieval metrics...")
        
        # Load annotations
        metrics_calc = EvaluationMetrics([a.__dict__ if hasattr(a, '__dict__') else a 
                                          for a in annotations])
        
        # Get results for each query
        query_results = {}
        for query in queries:
            results = self.pipeline.search(query.query_text, k=max(k_values))
            retrieved = [r['chunk_id'] for r in results]
            query_results[query.query_id] = retrieved
        
        # Compute overall metrics
        overall = metrics_calc.compute_overall_metrics(query_results, k_values=k_values)
        
        # Save metrics
        metrics_file = self.eval_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(overall, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Step 5: Save evaluation report
        report = {
            'summary': {
                'num_documents': len(pdf_paths),
                'num_chunks': len(chunks),
                'num_queries': len(queries),
                'query_type_distribution': {},
                'num_annotations': len(annotations)
            },
            'metrics': overall
        }
        
        # Add query type distribution
        for query in queries:
            qtype = query.query_type
            report['summary']['query_type_distribution'][qtype] = \
                report['summary']['query_type_distribution'].get(qtype, 0) + 1
        
        report_file = self.eval_dir / "report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report to {report_file}")
        
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        
        return report
    
    def print_report(self, report: Dict):
        """Print evaluation report."""
        summary = report['summary']
        metrics = report['metrics']
        
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        print("\nDocument Summary:")
        print(f"  Documents processed: {summary['num_documents']}")
        print(f"  Chunks created: {summary['num_chunks']}")
        print(f"  Queries generated: {summary['num_queries']}")
        print(f"  Annotations: {summary['num_annotations']}")
        
        print("\nQuery Type Distribution:")
        for qtype, count in summary['query_type_distribution'].items():
            print(f"  {qtype}: {count}")
        
        print("\nRetrieval Metrics:")
        print("\n  Precision, Recall, F1 by K:")
        for k in [1, 3, 5, 10]:
            print(f"    K={k}:")
            print(f"      P@{k}: {metrics.get(f'precision@{k}', 0):.3f}")
            print(f"      R@{k}: {metrics.get(f'recall@{k}', 0):.3f}")
            print(f"      F1@{k}: {metrics.get(f'f1@{k}', 0):.3f}")
        
        print(f"\n  Mean Reciprocal Rank (MRR): {metrics.get('mrr', 0):.3f}")
        print(f"  Mean Average Precision (MAP): {metrics.get('map', 0):.3f}")
        
        print(f"\n  Overall Precision@10: {metrics.get('overall_precision', 0):.3f}")
        print(f"  Overall Recall@10: {metrics.get('overall_recall', 0):.3f}")
        
        print("\n" + "=" * 60)
        
        # Check precision target
        precision_10 = metrics.get('overall_precision', 0)
        if precision_10 >= 0.9:
            print(f"✓ Precision target met ({precision_10:.1%} >= 90%)")
        else:
            print(f"✗ Precision below target ({precision_10:.1%} < 90%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fair Work RAG Evaluation Runner')
    parser.add_argument('--pdfs', nargs='+', required=True, help='PDF files to process')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of queries')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='data/processed', help='Output directory')
    parser.add_argument('--eval-dir', default='evaluation_set', help='Evaluation directory')
    parser.add_argument('--no-print', action='store_true', help='Skip printing report')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return
    
    # Initialize runner
    runner = EvaluationRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        eval_dir=args.eval_dir
    )
    
    # Run evaluation
    report = runner.run_evaluation(
        pdf_paths=args.pdfs,
        num_queries=args.num_queries
    )
    
    # Print report
    if not args.no_print:
        runner.print_report(report)


if __name__ == "__main__":
    main()
