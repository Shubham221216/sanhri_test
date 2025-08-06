
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from app.db import database
from sqlalchemy import text
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from datetime import date
import logging

router = APIRouter(prefix="/rag", tags=["RAG Assistant"])

# ✅ Request and Response models
class QueryInput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: list[str]

# ✅ Load model and prompt
model = ChatGoogleGenerativeAI(model='learnlm-2.0-flash-experimental')

prompt_template_str = """
You are SanHri-X, a friendly and intelligent virtual assistant designed to help customers with mall-related queries like directions, store timings, and current offers.

Use the information below to answer the user’s query. If no relevant data is found, say so politely.

⚠️ Format the answer in plain, spoken-friendly text. Avoid using symbols like '*', '-', or markdown. Present lists as sentences.

Context:
{context}

User Query:
{input}
"""


prompt = PromptTemplate.from_template(prompt_template_str)

# ✅ Helper: Get mall documents from DB
async def fetch_store_documents():
    await database.connect()
    query = text("""
        SELECT 
            s.shop_name,
            d.floor,
            s.category,
            h.day_of_week,
            h.open_time,
            h.close_time,
            h.is_closed,
            p.title AS offer
        FROM shops s
        LEFT JOIN store_operating_hours h ON s.shop_id = h.shop_id
        LEFT JOIN store_promotions p ON s.shop_id = p.shop_id
        LEFT JOIN store_directory d ON s.shop_id = d.store_id
        LIMIT 10
    """)
    rows = await database.fetch_all(query)
    await database.disconnect()

    return [
        Document(
            page_content=(
                f"{row['shop_name']} is located on floor {row['floor']}. "
                f"It belongs to the '{row['category']}' category. "
                f"On {row['day_of_week']}, it is "
                f"{'closed' if row['is_closed'] else f'open from {row['open_time']} to {row['close_time']}'}."
                f" Current offer: {row['offer'] or 'No current offers'}."
            )
        )
        for row in rows
    ]

# ✅ Health check
@router.get("/health")
async def health_check():
    return {"status": "ok", "date": date.today().isoformat()}

# ✅ Main RAG endpoint
@router.post("/ask", response_model=QueryResponse)
async def ask_virtual_assistant(query_input: QueryInput):
    query = query_input.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        docs = await fetch_store_documents()
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        final_docs = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(final_docs, embeddings)

        document_chain = create_stuff_documents_chain(model, prompt)
        retriever = vectorstore.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        result = retriever_chain.invoke({"input": query})

        return {
            "answer": result.get("answer") or result.get("output", "Sorry, I couldn't find a response."),
            "context": [doc.page_content for doc in result.get("context", [])]
        }

    except Exception as e:
        logging.exception("Error in RAG assistant:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
