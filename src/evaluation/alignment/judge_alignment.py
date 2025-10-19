#!/usr/bin/env python3
"""
Judge Alignment Harness

A robust CLI tool for evaluating how well automated judges align with human-labeled ground truth data.
Supports all binary judges: needs_handoff, groundedness, answer_relevance, summary_actionability, observability_coverage.

Usage:
    python src/evaluation/alignment/judge_alignment.py \
        --judge needs_handoff \
        --input-csv human_labels.csv \
        --output-json results.json

Input CSV format:
    trace_id,label
    session_00046e5f-2543-4569-87b4-715c35b64154,True
    session_0017b404-a278-4266-b9bd-fb6e602e6cd4,False
    ...

Output JSON format:
    {
        "tp": 15, "fp": 3, "tn": 12, "fn": 5,
        "precision": 0.833, "recall": 0.75, "f1_score": 0.789,
        "misclassified_trace_ids": ["trace1", "trace2"]
    }
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.judges.needs_handoff_judge import evaluate_needs_handoff
from evaluation.judges.groundedness_judge import evaluate_groundedness
from evaluation.judges.answer_relevance_judge import evaluate_answer_relevance
from evaluation.judges.summary_actionability_judge import evaluate_summary_actionability
from evaluation.judges.observability_coverage_judge import evaluate_observability_coverage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Judge function mapping
JUDGE_FUNCTIONS = {
    'needs_handoff': evaluate_needs_handoff,
    'groundedness': evaluate_groundedness,
    'answer_relevance': evaluate_answer_relevance,
    'summary_actionability': evaluate_summary_actionability,
    'observability_coverage': evaluate_observability_coverage
}

# Default trace directories
DEFAULT_TRACE_DIRS = [
    'context_engineering',
    'logs',
    'results'
]


class TraceLoader:
    """Loads and processes trace snapshots from various sources."""
    
    def __init__(self, trace_dirs: Optional[List[str]] = None):
        """Initialize with trace directories to search."""
        self.trace_dirs = trace_dirs or DEFAULT_TRACE_DIRS
        self.trace_cache = {}
        
    def load_trace_snapshot(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a trace snapshot by trace_id.
        
        Args:
            trace_id: The unique identifier for the trace (e.g., session UUID)
            
        Returns:
            PII-safe trace snapshot dict or None if not found
        """
        # Check cache first
        if trace_id in self.trace_cache:
            return self.trace_cache[trace_id]
            
        # Search for trace file
        trace_data = self._find_and_load_trace(trace_id)
        if trace_data:
            # Process raw trace into judge-friendly format
            snapshot = self._create_trace_snapshot(trace_data, trace_id)
            self.trace_cache[trace_id] = snapshot
            return snapshot
            
        logger.warning(f"Trace not found: {trace_id}")
        return None
    
    def _find_and_load_trace(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        """Find and load raw trace data from available sources."""
        for trace_dir in self.trace_dirs:
            if not os.path.exists(trace_dir):
                continue
                
            # Look for JSONL files containing the trace_id
            for file_path in Path(trace_dir).glob(f"*{trace_id}*"):
                if file_path.suffix in ['.jsonl', '.json']:
                    try:
                        return self._load_jsonl_file(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
                        continue
                        
            # Also check files without trace_id in name (batch files)
            for file_path in Path(trace_dir).glob("*.jsonl"):
                try:
                    data = self._load_jsonl_file(file_path)
                    if self._contains_trace_id(data, trace_id):
                        return data
                except Exception as e:
                    continue  # Skip problematic files
                    
        return None
    
    def _load_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file and return list of JSON objects."""
        events = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue
        return events
    
    def _contains_trace_id(self, events: List[Dict[str, Any]], trace_id: str) -> bool:
        """Check if any event contains the trace_id."""
        for event in events:
            if trace_id in str(event):
                return True
        return False
    
    def _create_trace_snapshot(self, raw_events: List[Dict[str, Any]], trace_id: str) -> Dict[str, Any]:
        """Convert raw trace events into judge-friendly snapshot format."""
        # Extract context stream events
        context_events = []
        summary_data = {}
        error_flags = False
        
        for event in raw_events:
            # Sanitize PII from event
            clean_event = self._sanitize_event(event)
            context_events.append(clean_event)
            
            # Check for error indicators
            if self._is_error_event(event):
                error_flags = True
                
            # Extract summary information
            if self._is_summary_event(event):
                summary_data.update(self._extract_summary_data(event))
        
        # Build snapshot in expected format
        snapshot = {
            'trace_id': trace_id,
            'context_stream': {
                'events': context_events,
                'summary': summary_data
            },
            'error_flags': error_flags,
            'metadata': {
                'total_events': len(context_events),
                'timestamp': self._get_latest_timestamp(raw_events)
            }
        }
        
        return snapshot
    
    def _sanitize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask PII from event data."""
        # Simple PII sanitization - can be enhanced based on requirements
        sanitized = {}
        
        for key, value in event.items():
            if isinstance(value, str):
                # Mask potential PII patterns (emails, phone numbers, etc.)
                sanitized[key] = self._mask_pii_patterns(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_event(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_event(item) if isinstance(item, dict) else item for item in value]
            else:
                sanitized[key] = value
                
        return sanitized
    
    def _mask_pii_patterns(self, text: str) -> str:
        """Mask common PII patterns in text."""
        import re
        
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers (basic patterns)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b\(\d{3}\)\s*\d{3}-\d{4}\b', '[PHONE]', text)
        
        # SSN patterns
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Credit card patterns (basic)
        text = re.sub(r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b', '[CREDITCARD]', text)
        
        return text
    
    def _is_error_event(self, event: Dict[str, Any]) -> bool:
        """Check if event indicates an error condition."""
        event_str = str(event).lower()
        error_indicators = ['error', 'failed', 'exception', 'timeout', 'rate_limit']
        return any(indicator in event_str for indicator in error_indicators)
    
    def _is_summary_event(self, event: Dict[str, Any]) -> bool:
        """Check if event contains summary information."""
        content_type = event.get('content_type', '')
        event_type = event.get('event_type', '')
        return any(keyword in content_type.lower() or keyword in event_type.lower() 
                  for keyword in ['summary', 'final', 'report', 'conclusion'])
    
    def _extract_summary_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary data from event."""
        summary = {}
        
        # Extract content as summary text
        if 'content' in event:
            summary['final_summary'] = event['content']
            
        # Extract other summary fields
        for key in ['executive_summary', 'summary', 'conclusions', 'recommendations']:
            if key in event:
                summary[key] = event[key]
                
        return summary
    
    def _get_latest_timestamp(self, events: List[Dict[str, Any]]) -> Optional[str]:
        """Get the latest timestamp from events."""
        timestamps = []
        for event in events:
            if 'timestamp' in event:
                timestamps.append(event['timestamp'])
        
        return max(timestamps) if timestamps else None


class AlignmentEvaluator:
    """Evaluates judge alignment with human labels."""
    
    def __init__(self, trace_loader: TraceLoader):
        self.trace_loader = trace_loader
    
    def evaluate_judge(self, judge_name: str, labeled_data: List[Tuple[str, bool]]) -> Dict[str, Any]:
        """
        Evaluate a judge against human-labeled data.
        
        Args:
            judge_name: Name of the judge to evaluate
            labeled_data: List of (trace_id, human_label) tuples
            
        Returns:
            Results dictionary with metrics and misclassified traces
        """
        if judge_name not in JUDGE_FUNCTIONS:
            raise ValueError(f"Unknown judge: {judge_name}. Available: {list(JUDGE_FUNCTIONS.keys())}")
        
        judge_function = JUDGE_FUNCTIONS[judge_name]
        
        # Confusion matrix counters
        tp = fp = tn = fn = 0
        misclassified_traces = []
        processing_errors = []
        
        logger.info(f"Evaluating {judge_name} on {len(labeled_data)} traces")
        
        for i, (trace_id, human_label) in enumerate(labeled_data):
            try:
                # Load trace snapshot
                snapshot = self.trace_loader.load_trace_snapshot(trace_id)
                if snapshot is None:
                    logger.warning(f"Skipping {trace_id}: trace not found")
                    processing_errors.append({
                        'trace_id': trace_id,
                        'error': 'trace_not_found'
                    })
                    continue
                
                # Run judge
                judge_prediction = judge_function(snapshot)
                
                # Update confusion matrix
                if human_label and judge_prediction:
                    tp += 1
                elif not human_label and not judge_prediction:
                    tn += 1
                elif not human_label and judge_prediction:
                    fp += 1
                    misclassified_traces.append({
                        'trace_id': trace_id,
                        'human_label': human_label,
                        'judge_prediction': judge_prediction,
                        'error_type': 'false_positive'
                    })
                else:  # human_label and not judge_prediction
                    fn += 1
                    misclassified_traces.append({
                        'trace_id': trace_id,
                        'human_label': human_label,
                        'judge_prediction': judge_prediction,
                        'error_type': 'false_negative'
                    })
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(labeled_data)} traces")
                    
            except Exception as e:
                logger.error(f"Error processing {trace_id}: {e}")
                processing_errors.append({
                    'trace_id': trace_id,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                continue
        
        # Calculate metrics
        metrics = self._calculate_metrics(tp, fp, tn, fn)
        
        # Compile results
        results = {
            'judge_name': judge_name,
            'total_traces': len(labeled_data),
            'processed_traces': tp + fp + tn + fn,
            'processing_errors': len(processing_errors),
            'confusion_matrix': {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            },
            'metrics': metrics,
            'misclassified_traces': [item['trace_id'] for item in misclassified_traces],
            'detailed_misclassifications': misclassified_traces,
            'processing_error_details': processing_errors
        }
        
        logger.info(f"Evaluation complete: {metrics}")
        return results
    
    def _calculate_metrics(self, tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        # Handle edge cases
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Additional metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1_score, 3),
            'accuracy': round(accuracy, 3),
            'specificity': round(specificity, 3)
        }


def load_labeled_csv(csv_path: str) -> List[Tuple[str, bool]]:
    """Load human-labeled CSV file."""
    labeled_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        if 'trace_id' not in reader.fieldnames or 'label' not in reader.fieldnames:
            raise ValueError("CSV must contain 'trace_id' and 'label' columns")
        
        for row_num, row in enumerate(reader, 2):  # Start at 2 for header
            trace_id = row['trace_id'].strip()
            label_str = row['label'].strip().lower()
            
            # Parse boolean label
            if label_str in ['true', '1', 'yes']:
                label = True
            elif label_str in ['false', '0', 'no']:
                label = False
            else:
                logger.warning(f"Invalid label '{label_str}' on row {row_num}, skipping")
                continue
            
            labeled_data.append((trace_id, label))
    
    logger.info(f"Loaded {len(labeled_data)} labeled traces from {csv_path}")
    return labeled_data


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate judge alignment with human-labeled ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--judge',
        required=True,
        choices=list(JUDGE_FUNCTIONS.keys()),
        help='Name of judge to evaluate'
    )
    
    parser.add_argument(
        '--input-csv',
        required=True,
        help='Path to CSV file with trace_id and label columns'
    )
    
    parser.add_argument(
        '--output-json',
        required=True,
        help='Path to write results JSON file'
    )
    
    parser.add_argument(
        '--trace-dirs',
        nargs='+',
        default=DEFAULT_TRACE_DIRS,
        help='Directories to search for trace files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load labeled data
        labeled_data = load_labeled_csv(args.input_csv)
        if not labeled_data:
            logger.error("No valid labeled data found")
            return 1
        
        # Initialize components
        trace_loader = TraceLoader(args.trace_dirs)
        evaluator = AlignmentEvaluator(trace_loader)
        
        # Run evaluation
        results = evaluator.evaluate_judge(args.judge, labeled_data)
        
        # Save results
        save_results_json(results, args.output_json)
        
        # Print summary
        metrics = results['metrics']
        print(f"\nJudge Alignment Results for '{args.judge}':")
        print(f"═══════════════════════════════════════")
        print(f"Processed: {results['processed_traces']}/{results['total_traces']} traces")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1_score']:.3f}")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Misclassified: {len(results['misclassified_traces'])} traces")
        
        if results['processing_errors'] > 0:
            print(f"⚠️  Processing errors: {results['processing_errors']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())