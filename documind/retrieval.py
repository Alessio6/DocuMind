import math
import re
from collections import Counter
from typing import Any, Literal, Protocol

from documind.documents import Chunk, Document


RetrievalMode = Literal["keyword", "vector", "hybrid"]
ABSTENTION_THRESHOLDS: dict[RetrievalMode, float] = {
	"keyword": 0.20,
	"vector": 0.35,
	"hybrid": 0.35,
}

STOP_WORDS = {
	"a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
	"be", "because", "been", "before", "between", "both", "but", "by", "can", "could",
	"did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
	"into", "is", "it", "its", "may", "might", "more", "most", "must", "of", "on", "or",
	"our", "should", "so", "than", "that", "the", "their", "then", "there", "these", "they",
	"this", "those", "to", "under", "up", "was", "we", "were", "what", "when", "which",
	"who", "will", "with", "would", "you", "your",
}
TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._+-]*")


class Encoder(Protocol):
	def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class RetrievalError(RuntimeError):
	"""Raised when semantic retrieval cannot be initialized or executed."""


def tokenize(text: str) -> list[str]:
	return [word.lower() for word in TOKEN_RE.findall(text) if word.lower() not in STOP_WORDS]


def _corpus(documents: list[Document]) -> list[Chunk]:
	return [chunk for document in documents for chunk in document["chunks"]]


def _keyword_scores(query: str, corpus: list[Chunk]) -> list[tuple[float, Chunk]]:
	query_terms = set(tokenize(query))
	if not query_terms or not corpus:
		return []
	tokenized = [tokenize(chunk["text"]) for chunk in corpus]
	document_frequency = Counter(term for terms in tokenized for term in set(terms))
	average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
	scored = []
	for chunk, terms in zip(corpus, tokenized):
		counts = Counter(terms)
		score = 0.0
		for term in query_terms:
			frequency = counts[term]
			if not frequency:
				continue
			inverse_frequency = math.log(
				1 + (len(corpus) - document_frequency[term] + 0.5)
				/ (document_frequency[term] + 0.5)
			)
			denominator = frequency + 1.5 * (
				1 - 0.75 + 0.75 * len(terms) / max(1, average_length)
			)
			score += inverse_frequency * frequency * 2.5 / denominator
		if score > 0:
			scored.append((score, chunk))
	return sorted(scored, key=lambda item: item[0], reverse=True)


def keyword_retrieve(
	query: str, documents: list[Document], limit: int = 5
) -> list[dict[str, Any]]:
	if limit <= 0 or not query.strip():
		return []
	ranked = _keyword_scores(query, _corpus(documents))[:limit]
	return [
		{**chunk, "source_id": f"S{rank}", "score": score / (score + 1)}
		for rank, (score, chunk) in enumerate(ranked, start=1)
	]


def should_abstain(mode: RetrievalMode, sources: list[dict[str, Any]]) -> bool:
	return not sources or sources[0]["score"] < ABSTENTION_THRESHOLDS[mode]


class SemanticRetriever:
	def __init__(
		self,
		documents: list[Document],
		encoder: Encoder,
		faiss_module: Any,
	) -> None:
		self.corpus = _corpus(documents)
		self.encoder = encoder
		self.faiss = faiss_module
		self.index = None
		self.embeddings = None
		if self.corpus:
			import numpy as np

			texts = [chunk["text"] for chunk in self.corpus]
			self.embeddings = np.ascontiguousarray(
				encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
				dtype="float32",
			)
			if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.corpus):
				raise RetrievalError("The embedding model returned an unexpected vector shape.")
			self.index = faiss_module.IndexFlatIP(self.embeddings.shape[1])
			self.index.add(self.embeddings)

	def _vector_scores(self, query: str) -> list[tuple[float, Chunk]]:
		if not self.corpus or self.index is None:
			return []
		import numpy as np

		query_vector = np.ascontiguousarray(
			self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True),
			dtype="float32",
		)
		scores, indexes = self.index.search(query_vector, len(self.corpus))
		return [
			(float(score), self.corpus[index])
			for score, index in zip(scores[0], indexes[0])
			if index >= 0
		]

	def retrieve(
		self, query: str, mode: RetrievalMode = "hybrid", limit: int = 5
	) -> list[dict[str, Any]]:
		if limit <= 0 or not query.strip():
			return []
		if mode == "keyword":
			ranked = _keyword_scores(query, self.corpus)
			confidences = [
				(score / (score + 1), chunk) for score, chunk in ranked
			]
		elif mode == "vector":
			ranked = self._vector_scores(query)
			confidences = ranked
		elif mode == "hybrid":
			keyword = _keyword_scores(query, self.corpus)
			vector = self._vector_scores(query)
			keyword_rank = {chunk["id"]: rank for rank, (_, chunk) in enumerate(keyword, 1)}
			vector_rank = {
				chunk["id"]: (rank, score)
				for rank, (score, chunk) in enumerate(vector, 1)
				if score >= 0.25
			}
			chunks = {chunk["id"]: chunk for chunk in self.corpus}
			corpus_order = {chunk["id"]: index for index, chunk in enumerate(self.corpus)}
			combined = []
			for chunk_id in set(keyword_rank) | set(vector_rank):
				semantic = vector_rank.get(chunk_id, (0, 0.0))[1]
				rank_score = (
					1 / (60 + keyword_rank[chunk_id]) if chunk_id in keyword_rank else 0
				) + (
					1 / (60 + vector_rank[chunk_id][0]) if chunk_id in vector_rank else 0
				)
				combined.append((rank_score, max(semantic, 0.0), chunks[chunk_id]))
			combined.sort(key=lambda item: (-item[0], corpus_order[item[2]["id"]]))
			ranked = [(rank_score, chunk) for rank_score, _, chunk in combined]
			confidences = [(semantic, chunk) for _, semantic, chunk in combined]
		else:
			raise ValueError(f"Unsupported retrieval mode: {mode}")

		confidence_by_id = {chunk["id"]: score for score, chunk in confidences}
		return [
			{
				**chunk,
				"source_id": f"S{rank}",
				"score": confidence_by_id.get(chunk["id"], score),
			}
			for rank, (score, chunk) in enumerate(ranked[:limit], start=1)
		]


def load_semantic_retriever(
	documents: list[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> SemanticRetriever:
	try:
		import faiss
		from sentence_transformers import SentenceTransformer
	except ImportError as error:
		raise RetrievalError(
			"Semantic retrieval dependencies are missing. Install requirements.txt."
		) from error
	try:
		encoder = SentenceTransformer(model_name)
		return SemanticRetriever(documents, encoder, faiss)
	except Exception as error:
		raise RetrievalError(
			f"Could not load the embedding model '{model_name}': {error}"
		) from error
