from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_file(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader

    )
    document= loader.load()
    return document

from typing import List 
from langchain.schema import Document
def filter_to_minmum_doc(docs: List[Document])-> List[Document]:
    minimal_docs: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    text_chunk=text_splitter.split_documents(minimal_docs)
    return text_chunk


def download_embeddings():
    model_name="BAAI/bge-large-en-v1.5"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name,
       
    )
    return embeddings

print("run successfully store_index.py.........")