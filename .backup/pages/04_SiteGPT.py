import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import asyncio
import sys
import concurrent.futures

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

def load_website(url):
    # Windows에서 Playwright를 사용하기 위해 별도 스레드에서 ProactorEventLoop 설정
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    loader = AsyncChromiumLoader([url])
    return loader.load()

if url:
    try:
        # 별도 스레드에서 실행하여 메인 이벤트 루프 충돌 방지
        with concurrent.futures.ThreadPoolExecutor() as executor:
            docs = executor.submit(load_website, url).result()
            
        transformed = html2text_transformer.transform_documents(docs)
        st.write(docs)
    except Exception as e:
        st.error(f"An error occurred: {e}")
