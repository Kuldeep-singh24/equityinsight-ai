import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- Load ENV ----------------

load_dotenv()

os.makedirs("vector_store", exist_ok=True)

file_path = "vector_store/faiss_store.pkl"

# ---------------- Streamlit UI ----------------

st.set_page_config(
    page_title="EquityInsight AI",
    page_icon="📈"
)

st.title("📈 EquityInsight AI")

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(
        f"URL {i+1}"
    )
    urls.append(url)

process_btn = st.sidebar.button(
    "Process URLs"
)

# ---------------- Load Groq Model ----------------

try:

    groq_api = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api:

        st.error(
            "GROQ_API_KEY missing in .env"
        )

        st.stop()

    llm = ChatGroq(
        groq_api_key=groq_api,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

except Exception as e:

    st.error(
        f"LLM Error:\n{e}"
    )

    st.stop()

# ---------------- Process URLs ----------------

if process_btn:

    urls = [
        u for u in urls
        if u.strip()
    ]

    if len(urls) == 0:

        st.warning(
            "Please enter URLs"
        )

    else:

        with st.spinner(
            "Processing articles..."
        ):

            try:

                loader = WebBaseLoader(
                    urls
                )

                data = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                docs = splitter.split_documents(
                    data
                )

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                vectorstore = FAISS.from_documents(
                    docs,
                    embeddings
                )

                with open(
                    file_path,
                    "wb"
                ) as f:

                    pickle.dump(
                        vectorstore,
                        f
                    )

                st.success(
                    "URLs processed successfully ✅"
                )

            except Exception as e:

                st.error(
                    f"Processing Error:\n{e}"
                )

# ---------------- Ask Question ----------------

st.header(
    "Ask Question"
)

query = st.text_input(
    "Enter your question"
)

if st.button(
    "Get Answer"
):

    if not query:

        st.warning(
            "Enter question first"
        )

    elif not os.path.exists(
        file_path
    ):

        st.warning(
            "Please process URLs first"
        )

    else:

        with st.spinner(
            "Generating answer..."
        ):

            try:

                with open(
                    file_path,
                    "rb"
                ) as f:

                    vectorstore = pickle.load(
                        f
                    )

                docs = vectorstore.similarity_search(
                    query,
                    k=4
                )

                context = "\n".join(
                    [
                        d.page_content
                        for d in docs
                    ]
                )

                prompt = f"""
                Use only the provided context.

                Context:
                {context}

                Question:
                {query}

                Answer:
                """

                response = llm.invoke(
                    prompt
                )

                st.subheader(
                    "Answer"
                )

                st.write(
                    response.content
                )

            except Exception as e:

                st.error(
                    f"Error:\n{e}"
                )