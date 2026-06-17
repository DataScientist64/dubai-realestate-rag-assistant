import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Dubai Real Estate Assistant", page_icon="🏢")

@st.cache_resource
def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    )

@st.cache_resource
def load_vector_store():
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

@st.cache_resource
def get_chat_model():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
        temperature=0.3
    )

def answer_question(question, vector_store, chat_model):
    relevant_chunks = vector_store.similarity_search(question, k=4)
    context_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    system_prompt = (
        "You are a helpful assistant that answers questions about Dubai real estate, "
        "tenancy law, and property market trends. Only answer using the context provided below. "
        "If the answer is not in the context, say you do not have that information. "
        "Be clear and concise.\n\n"
        f"Context:\n{context_text}"
    )

    response = chat_model.invoke([
        ("system", system_prompt),
        ("human", question)
    ])

    return response.content

st.title("🏢 Dubai Real Estate Assistant")
st.caption("Ask questions about Dubai tenancy law, rent regulations, and property market trends.")

vector_store = load_vector_store()
chat_model = get_chat_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about Dubai real estate...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(user_question, vector_store, chat_model)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
