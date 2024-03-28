from pathlib import Path

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import CharacterTextSplitter


def load_pdf(
    file_path: Path | str, start_page: int = 0, end_page: int = -1
) -> list[Document]:
    print(f"Loading PDF: {file_path}, start_page: {start_page}, end_page: {end_page}")
    loader = PyPDFLoader(str(file_path))
    return (
        loader.load()[start_page:end_page]
        if end_page != -1
        else loader.load()[start_page:]
    )


def load_text(file_path: Path | str) -> list[Document]:
    loader = TextLoader(str(file_path))
    return loader.load()


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    return text_splitter.split_documents(docs)
