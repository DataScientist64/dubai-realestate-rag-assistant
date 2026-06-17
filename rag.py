import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_documents():
    documents_folder = "documents"
    all_documents = []
    for filename in os.listdir(documents_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(documents_folder, filename)
            print(f"Loading: {filename}")
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            all_documents.extend(pages)
    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    )

def create_vector_store(chunks):
    embeddings = get_embeddings()
    print("\nCreating embeddings and building FAISS index... this may take a minute.")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("FAISS index saved locally as faiss_index folder.")
    return vector_store

def load_vector_store():
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

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

if __name__ == "__main__":
    if not os.path.exists("faiss_index"):
        documents = load_documents()
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)
    else:
        print("Loading existing FAISS index...")
        vector_store = load_vector_store()

    chat_model = get_chat_model()

    print("\nDubai Real Estate Assistant is ready. Type your question (or type exit to quit).\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = answer_question(question, vector_store, chat_model)
        print(f"\nAssistant: {answer}\n")
