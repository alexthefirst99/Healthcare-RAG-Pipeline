"""Part 2: LangChain RAG implementation.

Note: the assignment's sample code (`langchain.chains.RetrievalQA`, `langchain.document_loaders`,
etc.) targets LangChain ~0.1. A fresh `pip install langchain` today pulls 1.x, which removed
`langchain.chains` entirely in favor of LCEL-composed runnables. This is written against the
current (1.x) API - see the README for the version note.
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT = ChatPromptTemplate.from_template(
    "Answer the medical question using ONLY the provided context. "
    "If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


class LangChainRAG:
    def __init__(self, file_paths):
        docs = []
        for path in file_paths:
            docs.extend(PyPDFLoader(path).load())

        splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = FAISS.from_documents(splits, embeddings)
        print(f"LangChain: Indexed {len(splits)} chunks")

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        self.chain = PROMPT | self.llm | StrOutputParser()

    def retrieve(self, query):
        return self.retriever.invoke(query)

    def generate(self, query):
        docs = self.retrieve(query)
        context = "\n---\n".join(d.page_content for d in docs)
        answer = self.chain.invoke({"context": context, "question": query})
        return answer, [d.page_content for d in docs]
