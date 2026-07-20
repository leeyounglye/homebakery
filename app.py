import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------
# 환경변수
# ----------------------------------
load_dotenv(override=True)

# ----------------------------------
# Streamlit 설정
# ----------------------------------
st.set_page_config(
    page_title="홈 베이커리 챗봇",
    page_icon="👨‍🍳",
    layout="wide"
)

st.markdown(
    """
    <h1 style='font-size:25px;text-align:center;'>
    🍞 홈 베이커리 챗봇 🍰
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align:center;font-size:20px;color:gray;'>
    지도강사 : KHS 👨‍🍳
    </div>
    <br>
    """,
    unsafe_allow_html=True
)


# ----------------------------------
# LLM
# ----------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ----------------------------------
# Vector DB 생성 또는 로드
# ----------------------------------
@st.cache_resource
def load_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    DB_PATH = "faiss_db"

    # 이미 생성된 벡터DB 사용
    if os.path.exists(DB_PATH):

        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
 

    loader = TextLoader("data/홈베이커리.md", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)


    vectordb = FAISS.from_documents(
        docs,
        embeddings
    )

    vectordb.save_local(DB_PATH)

    return vectordb


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ----------------------------------
# 질문 처리 함수
# ----------------------------------
def ask_manual(question):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = ChatPromptTemplate.from_template(
        """
너는 홈 베이커리 수업을 담당하는 강사이다.
다음 참고문서를 기반으로 질문에 답변하라.

[참고문서]

{context}

[질문]

{question}

규칙

1. 한글로 답변
2. 간결하게 답변
3. 문서에 없는 내용은 추측하지 말 것
"""
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return result.content, docs

# ----------------------------------
# 채팅 기록 초기화
# ----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------
# 이전 채팅 출력
# ----------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------
# 입력창
# ----------------------------------
question = st.chat_input(
    "여기에 질문하세요... 예) 시오빵 레시피 알려줘"
)

# ----------------------------------
# 질문 처리
# ----------------------------------
if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🧁 레시피를 검색중입니다..."):
            answer, docs = ask_manual(question)
            st.markdown(answer)
            source_text = ""
            for i, doc in enumerate(docs):
                source_text += (
                    f"### 문서 {i+1}\n\n"
                    f"{doc.page_content[:500]}\n\n"
                )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )