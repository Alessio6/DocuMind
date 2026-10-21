from pathlib import Path

from documind.evaluate import DEFAULT_DATASET, _metrics, load_dataset
from documind.retrieval import should_abstain


def test_benchmark_has_at_least_50_labelled_questions_and_abstention_cases():
	data = load_dataset(DEFAULT_DATASET)

	assert len(data["questions"]) == 60
	assert sum(question["answerable"] for question in data["questions"]) == 50
	assert sum(not question["answerable"] for question in data["questions"]) == 10
	assert len(data["citation_examples"]) >= 8


def test_expected_answers_are_present_in_their_labelled_passages():
	data = load_dataset(DEFAULT_DATASET)
	chunks = {chunk["id"]: chunk["text"].casefold() for chunk in data["chunks"]}

	for question in data["questions"]:
		if question["answerable"]:
			assert any(
				question["expected_answer"].casefold() in chunks[chunk_id]
				for chunk_id in question["gold_chunk_ids"]
			), question["id"]


def test_metrics_count_abstentions_and_supported_citations():
	questions = [
		{"id": "known", "answerable": True, "gold_chunk_ids": ["gold"],
		 "expected_answer": "42"},
		{"id": "unknown", "answerable": False, "gold_chunk_ids": [],
		 "expected_answer": ""},
	]
	ranked = {
		"known": [{"id": "gold", "text": "The answer is 42.", "score": 0.9}],
		"unknown": [],
	}

	metrics, cases = _metrics(questions, ranked, "keyword")

	assert metrics["answer_correctness"] == 1.0
	assert metrics["citation_accuracy"] == 1.0
	assert metrics["abstention_precision"] == 1.0
	assert metrics["abstention_recall"] == 1.0
	assert len(cases) == 2


def test_malicious_document_and_unanswerable_attack_questions_are_labelled():
	data = load_dataset(DEFAULT_DATASET)
	malicious_chunk = next(chunk for chunk in data["chunks"] if chunk["id"] == "document-safety")
	attack_questions = [question for question in data["questions"] if question["id"] in {"q058", "q059", "q060"}]

	assert "Ignore all prior instructions" in malicious_chunk["text"]
	assert all(not question["answerable"] for question in attack_questions)
	assert all(should_abstain("keyword", []) for _ in attack_questions)
