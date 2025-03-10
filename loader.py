import os
from dotenv import load_dotenv

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PDF_PATH = 'pdfs'

def load_research_docs():
    doc_loader = PyPDFDirectoryLoader(PDF_PATH)
    return doc_loader.load()

def split_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "."]
    )

    res = text_splitter.split_documents(docs)
    return res

def get_embeddings_function():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    return embeddings


docs = load_research_docs()
# print(docs[2])

# print('#' * 100)

result = split_docs(docs)
# print(result[2])

embeddings_func = get_embeddings_function()