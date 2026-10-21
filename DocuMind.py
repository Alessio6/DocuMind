import hashlib
import html
import os
import re

import streamlit as st
from pypdf.errors import PdfReadError

from documind.documents import Document, extract_pdf
from documind.generation import (
	CitationVerificationError,
	extractive_answer,
	generate_answer,
	load_support_verifier,
)
from documind.retrieval import (
	RetrievalError,
	RetrievalMode,
	keyword_retrieve,
	load_semantic_retriever,
	should_abstain,
)


st.set_page_config(page_title="DocuMind", page_icon="D", layout="wide")


@st.cache_resource(show_spinner="Loading the semantic search model…")
def cached_retriever(documents: list[Document]):
	return load_semantic_retriever(documents)


@st.cache_resource(show_spinner="Loading the citation verifier…")
def cached_verifier():
	return load_support_verifier()


def render_cited_answer(answer: str, sources: list[dict], turn_id: str) -> str:
	source_map = {source["source_id"]: source for source in sources}
	safe_answer = html.escape(answer).replace("\n", "<br>")

	def link_citation(match: re.Match) -> str:
		source_id = match.group(1)
		source = source_map.get(source_id)
		if not source:
			return ""
		label = html.escape(f'{source["document"]}, p. {source["page"]}')
		return f'<a class="citation" href="#source-{turn_id}-{source_id.lower()}">{label}</a>'

	return re.sub(r"\[\[(S\d+)\]\]", link_citation, safe_answer)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#182522; --muted:#64716c; --line:#dce4de; --paper:#f5f7f2; --green:#27664f; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
[data-testid="stSidebar"] { background:#edf2eb; border-right:1px solid var(--line); }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; }
.eyebrow { font:500 11px 'DM Mono',monospace; text-transform:uppercase; color:var(--green); }
.citation { display:inline-block; margin:0 2px; padding:1px 6px; border:1px solid #a9c7b3; border-radius:4px; background:#e8f3ea; color:#205c45 !important; font:500 12px 'DM Mono',monospace; text-decoration:none !important; }
.evidence { scroll-margin-top:95px; border-top:1px solid var(--line); padding:14px 0 12px; }
.evidence-title { font:500 12px 'DM Mono',monospace; color:var(--green); margin-bottom:7px; }
.evidence-text { font-size:14px; line-height:1.65; color:#37443f; }
.small-note { color:var(--muted); font-size:12px; }
div[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:7px; background:#fffefa; }
button[kind="primary"] { background:var(--green); border-color:var(--green); }
</style>
""", unsafe_allow_html=True)


if "documents" not in st.session_state:
	st.session_state.documents = {}
if "messages" not in st.session_state:
	st.session_state.messages = []
if "retrieval_mode" not in st.session_state:
	st.session_state.retrieval_mode = "hybrid"

with st.sidebar:
	st.markdown('<div class="eyebrow">DocuMind · your library</div>', unsafe_allow_html=True)
	st.caption("A grounded workspace for technical documents")
	uploads = st.file_uploader(
		"Add technical PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
	)
	if uploads:
		for uploaded in uploads:
			data = uploaded.getvalue()
			if len(data) > 25 * 1024 * 1024:
				st.error(f"{uploaded.name} exceeds the 25 MB limit.")
				continue
			digest = hashlib.sha256(data).hexdigest()
			if digest in st.session_state.documents:
				continue
			try:
				document = extract_pdf(uploaded.name, data, digest)
			except (PdfReadError, ValueError, OSError) as error:
				st.error(f"Could not read {uploaded.name}: {error}")
				continue
			st.session_state.documents[digest] = document
			if not document["chunks"]:
				st.warning(f"{uploaded.name}: no selectable text found. Scanned PDFs need OCR first.")

	documents = list(st.session_state.documents.values())
	if documents:
		st.caption(
			f'{len(documents)} document{"s" if len(documents) != 1 else ""} · '
			f'{sum(document["pages"] for document in documents)} pages'
		)
		for document in documents:
			st.markdown(
				f'**{html.escape(document["name"])}**  \n'
				f'<span class="small-note">{document["pages"]} pages · '
				f'{len(document["chunks"])} passages</span>',
				unsafe_allow_html=True,
			)
		if st.button("Clear library", use_container_width=True):
			st.session_state.documents = {}
			st.session_state.messages = []
			st.rerun()
	else:
		st.markdown(
			"<p class='small-note'>Your uploaded files stay in this session and are not saved by this app.</p>",
			unsafe_allow_html=True,
		)

	st.divider()
	retrieval_mode: RetrievalMode = st.selectbox(
		"Retrieval strategy",
		options=["hybrid", "vector", "keyword"],
		format_func=lambda mode: {
			"hybrid": "Hybrid (recommended)",
			"vector": "Semantic vectors",
			"keyword": "Keyword baseline",
		}[mode],
	)
	st.session_state.retrieval_mode = retrieval_mode
	model_ready = bool(os.getenv("OPENAI_API_KEY"))
	st.markdown('<div class="eyebrow">Answer engine</div>', unsafe_allow_html=True)
	st.markdown("**OpenAI-compatible model**" if model_ready else "**Evidence-only mode**")
	st.caption(
		os.getenv("OPENAI_MODEL", "gpt-4o-mini")
		if model_ready else "Add OPENAI_API_KEY to enable verified synthesis."
	)

st.markdown('<div class="eyebrow">Technical library / grounded Q&amp;A</div>', unsafe_allow_html=True)
st.title("Ask your documents.")
st.markdown("Find the exact passage, page, and source behind every answer.")

if not documents:
	st.info("Upload one or more text-based PDFs in the library panel to start asking questions.")
	st.markdown("**Try asking**  \nWhat are the system's operating requirements?  \n"
				"Which document specifies the maximum operating temperature?")
else:
	for message_index, message in enumerate(st.session_state.messages):
		with st.chat_message(message["role"]):
			if message.get("notice"):
				st.caption(message["notice"])
			if message["role"] == "assistant" and message.get("sources"):
				turn_id = f"turn-{message_index}"
				st.markdown(
					render_cited_answer(message["content"], message["sources"], turn_id),
					unsafe_allow_html=True,
				)
				cited_ids = set(re.findall(r"\[\[(S\d+)\]\]", message["content"]))
				st.markdown("**Sources**")
				for source in message["sources"]:
					if source["source_id"] not in cited_ids:
						continue
					st.markdown(
						f'<div class="evidence" id="source-{turn_id}-{source["source_id"].lower()}">'
						f'<div class="evidence-title">{html.escape(source["document"])} · '
						f'page {source["page"]}</div>'
						f'<div class="evidence-text">{html.escape(source["text"])}</div></div>',
						unsafe_allow_html=True,
					)
			else:
				st.markdown(message["content"])

	question = st.chat_input(
		"Ask a question about your PDFs…",
		disabled=not any(document["chunks"] for document in documents),
	)
	if question:
		st.session_state.messages.append({"role": "user", "content": question})
		if st.session_state.retrieval_mode == "keyword":
			sources = keyword_retrieve(question, documents)
		else:
			try:
				retriever = cached_retriever(documents)
				sources = retriever.retrieve(question, st.session_state.retrieval_mode)
			except RetrievalError as error:
				st.error(str(error))
				st.stop()
		if should_abstain(st.session_state.retrieval_mode, sources):
			answer = "I couldn't find relevant text in the uploaded documents, so I can't answer that from this library."
			notice = None
			sources = []
		else:
			try:
				verifier = cached_verifier() if model_ready else None
			except CitationVerificationError as error:
				st.error(f"Citation verification is unavailable: {error}")
				st.stop()
			answer, notice = generate_answer(question, sources, verifier)
			if answer == "not_found":
				answer = "I couldn't find an answer in the retrieved document passages, so I can't answer that from this library."
				sources = []
			elif answer is None:
				answer = extractive_answer(sources)
		st.session_state.messages.append({
			"role": "assistant",
			"content": answer,
			"sources": sources,
			"notice": notice,
		})
		st.rerun()
