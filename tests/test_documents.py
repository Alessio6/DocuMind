from documind.documents import split_page


def test_split_page_keeps_overlap_and_all_text():
	text = " ".join(f"word{index}" for index in range(100))
	chunks = split_page(text, max_chars=100, overlap=20)

	assert len(chunks) > 1
	assert all(chunk for chunk in chunks)
	assert chunks[0][-10:] in chunks[1]
	assert chunks[-1].split()[-1] == "word99"


def test_split_page_ignores_empty_input():
	assert split_page(" \n\t ") == []
