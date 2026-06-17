# Dubai Real Estate Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Dubai tenancy law, rent regulations, and property market trends - grounded entirely in real Dubai Land Department documents and market reports, with zero hallucination outside the knowledge base.

Live App: https://dxb-realestate-rag.streamlit.app/

## What It Does

Instead of manually reading through hundreds of pages of property regulations and market reports, users can simply ask a question in plain English and get an instant, accurate, source-grounded answer.

Example questions it can answer:
- What are the rules for rent increases in Dubai?
- Which area in Dubai has the highest ROI for affordable apartments?
- What are tenant rights under Dubai tenancy law?

If a question falls outside its knowledge base, the assistant explicitly says so rather than generating a hallucinated answer - a deliberate design choice enforced through system prompting.

## Knowledge Base

The assistant is grounded in three real, publicly available documents:

| Document | Source |
|---|---|
| Dubai Tenancy Guide (Law No. 26 of 2007) | Dubai Land Department |
| Rental Increase Decree (Decree No. 43 of 2013) | Dubai Land Department |
| Dubai Sales Market Report 2025 | Bayut |

## Architecture

User Question
      -> Streamlit Chat Interface
      -> FAISS Vector Search (top-k retrieval)
      -> Retrieved Context + Question
      -> Azure OpenAI (GPT-4o-mini)
      -> Grounded Answer

Pipeline:
1. PDF documents are loaded and split into overlapping chunks (1000 chars, 200 overlap)
2. Each chunk is converted into a vector embedding using Azure OpenAI's text-embedding-3-small
3. Embeddings are stored and indexed locally using FAISS
4. On each user query, the most relevant chunks are retrieved via similarity search
5. Retrieved chunks + the user's question are passed to gpt-4o-mini via Azure OpenAI, using a system prompt that restricts answers strictly to the provided context
6. The response is rendered in a Streamlit chat interface

## Tech Stack

- LLM: Azure OpenAI (GPT-4o-mini)
- Embeddings: Azure OpenAI (text-embedding-3-small)
- Orchestration: LangChain
- Vector Store: FAISS
- Frontend: Streamlit
- Deployment: Streamlit Community Cloud
- Version Control: Git / GitHub

## Why RAG Instead of Fine-Tuning

Fine-tuning was deliberately ruled out for this use case. The knowledge base is small, document-based, and needs to stay easily updatable - RAG allows new regulations or market reports to be added without retraining anything. Fine-tuning teaches a model style and format, not facts; RAG was the architecturally correct choice here.

## Key Engineering Decisions

- Chunking strategy: 1000-character chunks with 200-character overlap to preserve context across chunk boundaries, particularly important for legal clauses that span paragraph breaks
- Grounded refusal: The system prompt explicitly instructs the model to say "I do not have that information" rather than hallucinate when a question falls outside the retrieved context - verified through testing with out-of-scope questions
- Caching: Streamlit's @st.cache_resource is used to avoid reloading the embedding model and vector store on every user interaction

## Run It Locally

git clone https://github.com/DataScientist64/dubai-realestate-rag-assistant.git
cd dubai-realestate-rag-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Create a .env file with your own Azure OpenAI credentials:

AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
AZURE_OPENAI_CHAT_API_VERSION=2025-01-01-preview
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15

Then run:

streamlit run app.py

## What's Next

- Expand the knowledge base with additional RERA circulars and area-specific market reports
- Add source citation display in the UI so users can see which document an answer came from
- Add conversation memory for multi-turn follow-up questions

## Author

Aditya Tribhuvan
MSc AI and Computer Science, University of Birmingham Dubai
LinkedIn: https://www.linkedin.com/in/aditya-tribhuvan/
GitHub: https://github.com/DataScientist64

## Demo Video

Watch a 2-3 minute walkthrough: https://www.loom.com/share/373a6deb41ed480a876f6072047e3366
