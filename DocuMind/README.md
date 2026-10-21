# DocuMind

DocuMind is a Streamlit PDF question-answering prototype with selectable keyword, semantic-vector, and hybrid retrieval. It keeps PDF extraction, retrieval, generation, and evaluation in separate modules.

## Retrieval and citation checks

- **Keyword** is the BM25-style baseline.
- **Vector** uses `sentence-transformers/all-MiniLM-L6-v2` embeddings and a FAISS inner-product index over normalized vectors.
- **Hybrid** combines keyword and vector rankings with reciprocal rank fusion.
- Runtime versions are pinned in `requirements.txt`, including CPU-only PyTorch for repeatable installation.
- OpenAI-compatible generation is optional. When enabled, each answer claim must cite a retrieved source and pass an NLI entailment check using `cross-encoder/nli-deberta-v3-small`. Failed checks show retrieved passages instead of the generated response.
- PDFs are treated as untrusted evidence, not instructions. The NLI check and prompt are safeguards, not a guarantee that model outputs are correct.

The first vector search and first generated answer download their respective model weights. Keep network access available for that initial use. Keyword mode does not load embeddings or need a model download.

## Run locally

Use Python 3.11, then install the pinned dependencies and start the app:

```powershell
py -3.11 --version
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m streamlit run DocuMind.py
```

These commands use the Python 3.11 launcher explicitly and do not activate the virtual environment, so they work when PowerShell script execution is disabled. If Python 3.11 is not installed, install it with `py install 3.11` first. The Docker image uses Python 3.12.

For generated answers, set `OPENAI_API_KEY`. Optionally set `OPENAI_BASE_URL` and `OPENAI_MODEL` for a compatible endpoint/model. Without an API key, the app shows evidence-only answers.

## Reproducible evaluation

The manually labelled fixture in `data/benchmark.json` has 60 questions (50 answerable and 10 unsupported), 20 passages including malicious instruction text, and 8 claim/source support labels. It is a controlled example benchmark, not a substitute for a representative evaluation set built from your own PDFs.

Run:

```powershell
python -m documind.evaluate
```

This compares keyword, vector, and hybrid retrieval and writes `results/benchmark.json` and `results/benchmark.csv`. Override paths with `--dataset`, `--json-output`, and `--csv-output`. The benchmark loads both the embedding model and NLI verifier, so first execution may download model weights.

Latest saved run on the bundled 60-question fixture:

| Method | Recall@5 | Answer correctness | Citation accuracy | Abstention F1 |
|---|---:|---:|---:|---:|
| Keyword | 0.920 | 0.740 | 0.673 | 0.533 |
| Vector | 1.000 | 0.860 | 0.843 | 0.526 |
| Hybrid | 0.980 | 0.820 | 0.854 | 0.455 |

The NLI verifier classified all 8 manually labelled claim/passage pairs correctly in this run. These small synthetic-set scores are smoke-test evidence, not production-quality estimates; expand and independently review the labels before drawing broader conclusions.

Metric definitions:

- **Recall@5**: answerable questions with at least one labelled gold passage in the top five, divided by answerable questions.
- **Answer correctness**: answered questions where the top passage contains the labelled expected answer phrase, divided by answerable questions. This is a deterministic evidence proxy, not an LLM judge.
- **Citation accuracy**: emitted top-passage citations that match a question's labelled gold passage, divided by all emitted citations. Unsupported questions that are incorrectly answered count as wrong citations.
- **Abstention precision/recall/F1/accuracy**: classification metrics for declining unsupported questions, using the fixed thresholds declared in `documind/retrieval.py`. Thresholds are not tuned on a separate test set and should be calibrated before production use.
- **Citation verification**: accuracy, precision, recall, and F1 of the NLI verifier against the eight manually labelled claim/passage examples.

## Tests

```powershell
python -m pytest
```

Unit tests use deterministic fake encoders and indexes; they do not download models. The benchmark itself exercises the installed Sentence Transformers and FAISS stack.

## Docker

```powershell
docker build -t documind .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY documind
```

The container targets Linux/Python 3.12. Model weights are downloaded at runtime and cached by the model libraries in the container's user cache.
