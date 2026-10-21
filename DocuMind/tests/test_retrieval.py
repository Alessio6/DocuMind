from typing import Any

import numpy as np

from documind.retrieval import (
	ABSTENTION_THRESHOLDS,
	SemanticRetriever,
	keyword_retrieve,
	should_abstain,
)


class FakeEncoder:
	def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
		vectors = []
		for sentence in sentences:
			text = sentence.lower()
			if "cooling unit" in text or "thermal operating limits" in text:
				vectors.append([1.0, 0.0])
			else:
				vectors.append([0.0, 1.0])
		return np.asarray(vectors, dtype="float32")


class FakeIndex:
	def __init__(self, dimension: int) -> None:
		self.vectors = np.empty((0, dimension), dtype="float32")

	def add(self, vectors: np.ndarray) -> None:
		self.vectors = vectors

	def search(self, query: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
		scores = query @ self.vectors.T
		indexes = np.argsort(-scores, axis=1)[:, :count]
		return np.take_along_axis(scores, indexes, axis=1), indexes


class FakeFaiss:
	IndexFlatIP = FakeIndex


def sample_documents() -> list[dict[str, Any]]:
	return [{
		"id": "doc",
		"name": "manual.pdf",
		"pages": 1,
		"chunks": [
			{"id": "target", "document": "manual.pdf", "page": 1,
			 "text": "The cooling unit operates from 5 to 40 degrees Celsius."},
			{"id": "other", "document": "manual.pdf", "page": 1,
			 "text": "The network switch uses port 8443 for administration."},
		],
	}]


def test_keyword_baseline_and_abstention_for_no_match():
	docs = sample_documents()
	assert keyword_retrieve("cooling operating", docs)[0]["id"] == "target"
	assert keyword_retrieve("unrelated moon rocks", docs) == []
	assert should_abstain("keyword", [])
	assert ABSTENTION_THRESHOLDS["keyword"] > 0


def test_vector_retrieval_finds_a_semantic_paraphrase_missed_by_keywords():
	docs = sample_documents()
	query = "thermal operating limits"
	assert keyword_retrieve(query, docs) == []

	retriever = SemanticRetriever(docs, FakeEncoder(), FakeFaiss())
	sources = retriever.retrieve(query, "vector", limit=2)

	assert sources[0]["id"] == "target"
	assert sources[0]["source_id"] == "S1"
	assert sources[0]["score"] == 1.0


def test_hybrid_preserves_relevant_semantic_candidate():
	docs = sample_documents()
	retriever = SemanticRetriever(docs, FakeEncoder(), FakeFaiss())
	sources = retriever.retrieve("thermal operating limits", "hybrid")

	assert sources[0]["id"] == "target"
	assert sources[0]["score"] == 1.0


def test_hybrid_breaks_tied_scores_by_corpus_order():
	class TiedEncoder:
		def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
			return np.asarray([[1.0, 0.0] for _ in sentences], dtype="float32")

	retriever = SemanticRetriever(sample_documents(), TiedEncoder(), FakeFaiss())

	assert [source["id"] for source in retriever.retrieve("unknown words", "hybrid")] == [
		"target", "other",
	]
