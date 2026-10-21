import re
from io import BytesIO
from typing import TypedDict

from pypdf import PdfReader


class Chunk(TypedDict):
	id: str
	document: str
	page: int
	text: str


class Document(TypedDict):
	id: str
	name: str
	pages: int
	chunks: list[Chunk]


def split_page(text: str, max_chars: int = 1250, overlap: int = 180) -> list[str]:
	text = re.sub(r"\s+", " ", text).strip()
	if not text:
		return []
	chunks = []
	start = 0
	while start < len(text):
		end = min(start + max_chars, len(text))
		if end < len(text):
			boundary = text.rfind(" ", start + max_chars // 2, end)
			if boundary > start:
				end = boundary
		chunk = text[start:end].strip()
		if chunk:
			chunks.append(chunk)
		if end == len(text):
			break
		start = max(start + 1, end - overlap)
	return chunks


def extract_pdf(file_name: str, data: bytes, digest: str) -> Document:
	reader = PdfReader(BytesIO(data), strict=False)
	if reader.is_encrypted and not reader.decrypt(""):
		raise ValueError("This PDF is encrypted. Upload an unlocked copy.")
	chunks: list[Chunk] = []
	for page_number, page in enumerate(reader.pages, start=1):
		page_text = page.extract_text() or ""
		for chunk_number, text in enumerate(split_page(page_text), start=1):
			chunks.append({
				"id": f"{digest[:8]}-p{page_number}-c{chunk_number}",
				"document": file_name,
				"page": page_number,
				"text": text,
			})
	return {"id": digest, "name": file_name, "pages": len(reader.pages), "chunks": chunks}
