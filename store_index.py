from dotenv import load_dotenv
import os
from src.helper import load_pdf_file,filter_to_minmum_doc,text_split,download_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone import Pinecone


load_dotenv()

api_key = os.getenv("groq_api_key")
os.environ["GROQ_API_KEY"] = api_key
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

extracted_data=load_pdf_file("E:\cht2\data")
minimal_docs=filter_to_minmum_doc(extracted_data)
text_chunk=text_split(minimal_docs)


embedding=download_embeddings()

pine_cone_api=PINECONE_API_KEY
pc=Pinecone(api_key=pine_cone_api)

index_name="quickstart"


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")

    )


docsearch=PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)

