import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from documind.documents import Document
from documind.generation import CitationVerificationError, load_support_verifier
from documind.retrieval import (
	ABSTENTION_THRESHOLDS,
	RetrievalError,
	keyword_retrieve,
	load_semantic_retriever,
	should_abstain,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "benchmark.json"
METHODS = ("keyword", "vector", "hybrid")


def load_dataset(path: Path) -> dict[str, Any]:
	data = json.loads(path.read_text(encoding="utf-8"))
	if len(data["questions"]) < 50:
		raise ValueError("The labelled benchmark must contain at least 50 questions.")
	chunk_ids = {chunk["id"] for chunk in data["chunks"]}
	for question in data["questions"]:
		if not set(question["gold_chunk_ids"]).issubset(chunk_ids):
			raise ValueError(f'Question {question["id"]} references an unknown gold chunk.')
	return data


def _metrics(
	questions: list[dict[str, Any]],
	ranked: dict[str, list[dict[str, Any]]],
	mode: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
	answerable = [question for question in questions if question["answerable"]]
	recall_hits = 0
	correct_answers = 0
	correct_citations = 0
	emitted_citations = 0
	true_positive = false_positive = false_negative = true_negative = 0
	results = []

	for question in questions:
		sources = ranked[question["id"]]
		gold_ids = set(question["gold_chunk_ids"])
		hit_at_5 = any(source["id"] in gold_ids for source in sources[:5])
		if question["answerable"] and hit_at_5:
			recall_hits += 1
		predicted_answerable = not should_abstain(mode, sources)
		if question["answerable"]:
			if predicted_answerable and sources:
				top_source = sources[0]
				is_correct = (
					top_source["id"] in gold_ids
					and question["expected_answer"].casefold() in top_source["text"].casefold()
				)
				if is_correct:
					correct_answers += 1
			if predicted_answerable:
				emitted_citations += 1
				if sources[0]["id"] in gold_ids:
					correct_citations += 1
			if predicted_answerable:
				true_positive += 1
			else:
				false_negative += 1
		elif predicted_answerable:
			emitted_citations += 1
			false_positive += 1
		else:
			true_negative += 1

		results.append({
			"question_id": question["id"],
			"method": mode,
			"predicted_answerable": predicted_answerable,
			"gold_chunk_ids": sorted(gold_ids),
			"top5_chunk_ids": [source["id"] for source in sources[:5]],
			"top1_score": sources[0]["score"] if sources else 0.0,
			"recall_at_5_hit": hit_at_5,
		})

	recall = true_positive / max(1, true_positive + false_negative)
	precision = true_positive / max(1, true_positive + false_positive)
	abstention_precision = true_negative / max(1, true_negative + false_negative)
	abstention_recall = true_negative / max(1, true_negative + false_positive)
	metrics = {
		"recall_at_5": recall_hits / max(1, len(answerable)),
		"answer_correctness": correct_answers / max(1, len(answerable)),
		"citation_accuracy": correct_citations / max(1, emitted_citations),
		"answerability_precision": precision,
		"answerability_recall": recall,
		"abstention_precision": abstention_precision,
		"abstention_recall": abstention_recall,
		"abstention_f1": (
			2 * abstention_precision * abstention_recall
			/ max(1e-12, abstention_precision + abstention_recall)
		),
		"abstention_accuracy": (
			true_positive + true_negative
		) / max(1, len(questions)),
	}
	return metrics, results


def _citation_metrics(
	examples: list[dict[str, Any]], verifier: Any
) -> dict[str, float]:
	true_positive = false_positive = false_negative = true_negative = 0
	for example in examples:
		predicted = verifier.supports(example["claim"], example["passage"])
		expected = example["supported"]
		if predicted and expected:
			true_positive += 1
		elif predicted:
			false_positive += 1
		elif expected:
			false_negative += 1
		else:
			true_negative += 1
	precision = true_positive / max(1, true_positive + false_positive)
	recall = true_positive / max(1, true_positive + false_negative)
	return {
		"accuracy": (true_positive + true_negative) / max(1, len(examples)),
		"precision": precision,
		"recall": recall,
		"f1": 2 * precision * recall / max(1e-12, precision + recall),
		"labelled_examples": len(examples),
	}


def run_benchmark(data: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
	document: Document = {
		"id": "benchmark",
		"name": "DocuMind labelled benchmark",
		"pages": 0,
		"chunks": data["chunks"],
	}
	retriever = load_semantic_retriever([document])
	questions = data["questions"]
	ranked_by_method: dict[str, dict[str, list[dict[str, Any]]]] = {}
	method_metrics = {}
	case_results = []
	for method in METHODS:
		ranked = {}
		for question in questions:
			if method == "keyword":
				sources = keyword_retrieve(question["question"], [document], limit=5)
			else:
				sources = retriever.retrieve(question["question"], method, limit=5)
			ranked[question["id"]] = sources
		ranked_by_method[method] = ranked
		metrics, cases = _metrics(questions, ranked, method)
		metrics["abstention_threshold"] = ABSTENTION_THRESHOLDS[method]
		method_metrics[method] = metrics
		case_results.extend(cases)

	verifier = load_support_verifier()
	return {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"dataset": str(dataset_path.resolve().relative_to(ROOT))
		if dataset_path.resolve().is_relative_to(ROOT)
		else dataset_path.name,
		"question_count": len(questions),
		"answerable_question_count": sum(question["answerable"] for question in questions),
		"retrieval_model": "sentence-transformers/all-MiniLM-L6-v2",
		"citation_model": "cross-encoder/nli-deberta-v3-small",
		"citation_verification": _citation_metrics(data["citation_examples"], verifier),
		"methods": method_metrics,
		"cases": case_results,
	}


def write_results(results: dict[str, Any], json_path: Path, csv_path: Path) -> None:
	json_path.parent.mkdir(parents=True, exist_ok=True)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
	with csv_path.open("w", newline="", encoding="utf-8") as output:
		writer = csv.DictWriter(
			output,
			fieldnames=[
				"method", "recall_at_5", "answer_correctness", "citation_accuracy",
				"answerability_precision", "answerability_recall",
				"abstention_precision", "abstention_recall", "abstention_f1",
				"abstention_accuracy", "abstention_threshold",
			],
		)
		writer.writeheader()
		for method, metrics in results["methods"].items():
			writer.writerow({"method": method, **metrics})


def main() -> int:
	parser = argparse.ArgumentParser(description="Compare DocuMind retrieval and citation verification.")
	parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
	parser.add_argument("--json-output", type=Path, default=ROOT / "results" / "benchmark.json")
	parser.add_argument("--csv-output", type=Path, default=ROOT / "results" / "benchmark.csv")
	args = parser.parse_args()
	try:
		data = load_dataset(args.dataset)
		results = run_benchmark(data, args.dataset)
		write_results(results, args.json_output, args.csv_output)
	except (
		OSError, json.JSONDecodeError, KeyError, ValueError, RetrievalError,
		CitationVerificationError,
	) as error:
		print(f"Benchmark failed: {error}", file=sys.stderr)
		return 1
	print(f"Saved benchmark results to {args.json_output} and {args.csv_output}")
	for method, metrics in results["methods"].items():
		print(
			f'{method}: Recall@5={metrics["recall_at_5"]:.3f}, '
			f'answer correctness={metrics["answer_correctness"]:.3f}, '
			f'citation accuracy={metrics["citation_accuracy"]:.3f}, '
			f'abstention F1={metrics["abstention_f1"]:.3f}'
		)
	print(f'Citation support classification: {results["citation_verification"]}')
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
