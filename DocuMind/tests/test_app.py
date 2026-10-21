from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "DocuMind.py"


def test_streamlit_app_starts_without_runtime_errors():
	app = AppTest.from_file(str(APP_FILE)).run()

	assert not app.exception
	assert [title.value for title in app.title] == ["Ask your documents."]
