from dotenv import load_dotenv
from openrouter import embeddings

load_dotenv()
import pprint
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openrouter import ChatOpenRouter

# urls = ['https://lilianweng.github.io/posts/2023-06-23-agent/',
#         'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
#         'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/']
#
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
#
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=250, chunk_overlap=0
# )
#
# docs_splits = text_splitter.split_documents(docs_list)

embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

# vectorstore = Chroma.from_documents(
#     documents=docs_splits,
#     collection_name="rag-chroma",
#     embedding=embedding,
#     persist_directory="./.chroma"
# )

retriever = Chroma(
    collection_name="rag-chroma",
    embedding_function=embedding,
    persist_directory="./.chroma"
).as_retriever()
