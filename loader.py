import os
from dotenv import load_dotenv
import uuid

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.evaluation import load_evaluator
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

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

def create_vectorstore(chunks, embeddings_function, vectorstore_path):
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.page_content)) for chunk in chunks]

    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    vectorstore = Chroma.from_documents(documents=unique_chunks,
                                        ids=list(unique_ids),
                                        embedding=embeddings_function,
                                        persist_directory=vectorstore_path)
    
    vectorstore.persist()

docs = load_research_docs()

chunks = split_docs(docs)

embeddings_function = get_embeddings_function()

create_vectorstore(chunks, embeddings_function, vectorstore_path="vectorstore_chroma")

vectorstore = Chroma(persist_directory="vectorstore_chroma", embedding_function=embeddings_function)

retriever = vectorstore.as_retriever(search_type="similarity")
question = "What is Retrieval Augmented Generation?"
similar_chunks = retriever.invoke(question)


PROMPT_TEMPLATE = """
You are an assistant that has knowledge on research papers who will be answering related questions. 
Use the following chunks retrived from the context to answer the questions. If something is asked out of the context, 
use your pre-trained knowledge to answer.

{context}

----

Answer the question based on the above context: {question}
"""

context = "\n\n---\n\n".join([doc.page_content for doc in similar_chunks])

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context, question=question)


llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
answer = llm.invoke(prompt)

print('#' * 200)
print('\n')
print(answer.content)

# evaluator = load_evaluator(evaluator="embedding_distance", embeddings=embeddings_function)
