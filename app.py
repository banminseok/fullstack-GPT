from langchain.document_loaders import UnstructuredFileLoader
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

# """
# Stuff Documents 체인을 사용하여 완전한 RAG 파이프라인을 구현하세요.
# 체인을 수동으로 구현해야 합니다.
# 체인에 ConversationBufferMemory를 부여합니다.
# 이 문서를 사용하여 RAG를 수행하세요: https://gist.github.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223
# 체인에 다음 질문을 합니다:
# Aaronson 은 유죄인가요?
# 그가 테이블에 어떤 메시지를 썼나요?
# Julia 는 누구인가요?
# """
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message = f"{self.message}{token}"
        self.message_box.markdown(self.message + "▌")


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, apiKey):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 1. 문서 로드 (Document Loading)
    loader = UnstructuredFileLoader(file_path, encoding="utf-8")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    # 3. 임베딩 생성 및 캐시 (OpenAIEmbeddings, CacheBackedEmbeddings)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    embeddings = OpenAIEmbeddings(openai_api_key=apiKey)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    # 4. 벡터 스토어 생성 (FAISS)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def map_docs(inputs):
    documents = inputs["documents"]
    question = inputs["question"]
    # return "\n\n".join(
    #     map_doc_chain.invoke(
    #         {"context": doc.page_content, "question": question}
    #     ).content
    #     for doc in documents
    # )
    # 각 문서를 개별적으로 LLM에게 질문하여 관련 텍스트 추출
    relevant_texts = []
    for doc in documents:
        response = map_doc_chain.invoke(
            {"context": doc.page_content, "question": question}
        )
        relevant_texts.append(response.content)

    # 추출된 텍스트들을 합쳐서 반환
    rtnVal = "\n\n".join(relevant_texts)
    return rtnVal


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
사이드바에서 파일을 업로드하세요.

이 챗봇을 사용해 파일 관련 질문을 AI에게 물어보세요!

단 , OPENAI_API_KEY 가 필요합니다.
"""
)

# 6. 체인 연결
map_doc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
                -------
                {context}
                """,
        ),
        ("human", "{question}"),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given the following extracted parts of a long document and a question, create a final answer. 
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            please answer in Korean.
            ------
            context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

# --sidebar---------------------------------------------------
with st.sidebar:
    apiKey = st.text_input("OPENAI_API_KEY 입력")
    apiKey = apiKey.strip()
    if apiKey:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )
# --sidebar---------------------------------------------------

if apiKey:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=apiKey,
    )
    pllm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=apiKey,
    )

    if file:
        retriever = embed_file(file, apiKey)
        send_message(
            "선택한 파일을 읽어 임베딩을 생성했어요. 질문해주세요!",
            "ai",
            save=False,
        )
        paint_history()

        map_doc_chain = map_doc_prompt | pllm
        map_chain = {
            "documents": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(map_docs)

        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": map_chain,  # map_chain이 문맥 추출을 담당
                    "question": RunnablePassthrough(),  # 질문 그대로 전달
                }
                | final_prompt
                | llm
            )
            # chain = (
            #     {
            #         "context": retriever | RunnableLambda(format_docs),
            #         "question": RunnablePassthrough(),
            #     }
            #     | final_prompt
            #     | llm
            # )
            with st.chat_message("ai"):
                response = chain.invoke(message)
    else:
        st.session_state["messages"] = []
else:
    st.stop()
