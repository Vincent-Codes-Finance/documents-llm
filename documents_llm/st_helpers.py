from pathlib import Path

import streamlit as st

from .document import load_pdf
from .query import query_document
from .summarize import summarize_document


def save_uploaded_file(
    uploaded_file: "UploadedFile", output_dir: Path = Path("/tmp")
) -> Path:
    output_path = Path(output_dir) / uploaded_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path


def run_query(
    uploaded_file: "UploadedFile",
    summarize: bool,
    user_query: str,
    start_page: int,
    end_page: int,
    model_name: str,
    openai_api_key: str,
    openai_url: str,
    temperature: float,
) -> str:
    # Saves the uploaded file to a temporary location, loads the PDF, and deletes the file
    st.write("Saving the uploaded file...")
    file_path = save_uploaded_file(uploaded_file, output_dir=Path("/tmp"))
    st.write("Loading the document...")
    docs = load_pdf(file_path, start_page=start_page, end_page=end_page)
    file_path.unlink()

    if summarize:
        st.write("Summarizing the document...")
        return summarize_document(
            docs,
            model_name=model_name,
            openai_api_key=openai_api_key,
            base_url=openai_url,
            temperature=temperature,
        )
    st.write("Querying the document...")
    return query_document(
        docs,
        user_query=user_query,
        model_name=model_name,
        openai_api_key=openai_api_key,
        base_url=openai_url,
        temperature=temperature,
    )
