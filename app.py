from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

"""
Stuff Documents 체인을 사용하여 완전한 RAG 파이프라인을 구현하세요.
체인을 수동으로 구현해야 합니다.
체인에 ConversationBufferMemory를 부여합니다.
이 문서를 사용하여 RAG를 수행하세요: https://gist.github.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223
체인에 다음 질문을 합니다:
Aaronson 은 유죄인가요?
그가 테이블에 어떤 메시지를 썼나요?
Julia 는 누구인가요?
"""
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    topic = st.text_input("Search Wikipedia...")

llm = ChatOpenAI(
    temperature=0.1,
)
