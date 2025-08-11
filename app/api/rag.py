
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
from datetime import date, datetime, timedelta
import logging
import re
import asyncio
from typing import Optional

router = APIRouter(prefix="/rag", tags=["RAG Assistant"])

# --- Request and Response models
class QueryInput(BaseModel):
    query: str
    language: str  # 'en', 'hi', or 'mr'

class QueryResponse(BaseModel):
    answer: str
    context: list[str]

# --- Global cache for performance optimization
_vectorstore_cache: Optional[FAISS] = None
_cache_timestamp: Optional[datetime] = None
_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
_model: Optional[ChatGoogleGenerativeAI] = None
_documents_cache: list[Document] = []

# Cache duration (15 minutes)
CACHE_DURATION = timedelta(minutes=15)

# --- Initialize model with stable configuration
def get_model():
    global _model
    if _model is None:
        try:
            # Use stable Gemini model instead of experimental
            _model = ChatGoogleGenerativeAI(
                model='gemini-1.5-flash-8b',  # Stable model
                temperature=0.3,
                max_tokens=1024,
                timeout=30,  # 30 second timeout
                max_retries=2  # Reduce retries
            )
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {e}")
            # Fallback to another model if available
            _model = ChatGoogleGenerativeAI(
                model='gemini-pro',
                temperature=0.3,
                max_tokens=1024,
                timeout=30,
                max_retries=2
            )
    return _model

# --- Initialize embeddings
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            task_type="retrieval_document"
        )
    return _embeddings

# --- Query classification patterns
GENERAL_PATTERNS = {
    'en': [
        r'\b(hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening)\b',
        r'\b(how\s+are\s+you|what\'s\s+up|wassup)\b',
        r'\b(thank\s+you|thanks|bye|goodbye|see\s+you)\b',
        r'\b(who\s+are\s+you|what\s+is\s+your\s+name)\b',
        r'\b(help|assist|support)\b$'  # Single word help
    ],
    'hi': [
        r'\b(हैलो|हाय|नमस्ते|नमस्कार|सुप्रभात|शुभ\s+दिन)\b',
        r'\b(कैसे\s+हैं|क्या\s+हाल)\b',
        r'\b(धन्यवाद|शुक्रिया|अलविदा|नमस्ते)\b',
        r'\b(आप\s+कौन\s+हैं|आपका\s+नाम\s+क्या\s+है)\b'
    ],
    'mr': [
        r'\b(हॅलो|हाय|नमस्कार|नमस्ते|सुप्रभात|शुभ\s+दिन)\b',
        r'\b(कसे\s+आहात|काय\s+चालू\s+आहे)\b',
        r'\b(धन्यवाद|निरोप|अलविदा)\b',
        r'\b(तुम्ही\s+कोण\s+आहात|तुमचे\s+नाव\s+काय)\b'
    ]
}

# --- Check if query is general conversation
def is_general_query(query: str, language: str) -> bool:
    query_lower = query.lower().strip()
    patterns = GENERAL_PATTERNS.get(language, GENERAL_PATTERNS['en'])
    
    for pattern in patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    
    # Additional simple checks
    if len(query_lower.split()) <= 2 and any(word in query_lower for word in 
        ['hello', 'hi', 'hey', 'नमस्ते', 'हैलो', 'हाय', 'नमस्कार']):
        return True
        
    return False

# --- Generate quick responses for general queries
def get_general_response(query: str, language: str) -> str:
    query_lower = query.lower().strip()
    
    if language == "hi":
        if re.search(r'\b(हैलो|हाय|नमस्ते|नमस्कार)\b', query_lower):
            return "नमस्ते! मैं SanHri-X हूं, आपका मॉल असिस्टेंट। मैं आपकी दुकानों, समय, दिशा और ऑफ़र्स के बारे में मदद कर सकता हूं।"
        elif re.search(r'\b(धन्यवाद|शुक्रिया)\b', query_lower):
            return "आपका स्वागत है! क्या मैं आपकी और कोई मदद कर सकता हूं?"
        elif re.search(r'\b(अलविदा|बाय)\b', query_lower):
            return "अलविदा! मॉल में आपका दिन शुभ हो!"
        elif re.search(r'\b(कैसे\s+हैं|क्या\s+हाल)\b', query_lower):
            return "मैं बिल्कुल ठीक हूं और आपकी सेवा के लिए तैयार हूं! आप मॉल के बारे में कुछ भी पूछ सकते हैं।"
        elif re.search(r'\b(आप\s+कौन|आपका\s+नाम)\b', query_lower):
            return "मैं SanHri-X हूं, आपका स्मार्ट मॉल असिस्टेंट। मैं यहां स्टोर की जानकारी, समय और दिशा-निर्देश देने के लिए हूं।"
    
    elif language == "mr":
        if re.search(r'\b(हॅलो|हाय|नमस्कार|नमस्ते)\b', query_lower):
            return "नमस्कार! मी SanHri-X आहे, तुमचा मॉल असिस्टंट. मी तुम्हाला दुकाने, वेळा, दिशा आणि ऑफर्स बद्दल मदत करू शकतो."
        elif re.search(r'\b(धन्यवाद)\b', query_lower):
            return "कृपया! मी तुमची आणखी काही मदत करू शकतो का?"
        elif re.search(r'\b(निरोप|अलविदा)\b', query_lower):
            return "निरोप! मॉलमध्ये तुमचा दिवस चांगला जाईल!"
        elif re.search(r'\b(कसे\s+आहात|काय\s+चालू)\b', query_lower):
            return "मी पूर्णपणे तयार आहे आणि तुमची सेवा करण्यासाठी उत्सुक आहे! तुम्ही मॉल संबंधी काहीही विचारू शकता."
        elif re.search(r'\b(तुम्ही\s+कोण|तुमचे\s+नाव)\b', query_lower):
            return "मी SanHri-X आहे, तुमचा स्मार्ट मॉल असिस्टंट. मी येथे स्टोअरची माहिती, वेळ आणि दिशा-निर्देश देण्यासाठी आहे."
    
    else:  # English
        if re.search(r'\b(hello|hi|hey)\b', query_lower):
            return "Hello! I'm SanHri-X, your mall assistant. I can help you with store information, timings, directions, and current offers."
        elif re.search(r'\b(thank\s+you|thanks)\b', query_lower):
            return "You're welcome! Is there anything else I can help you with?"
        elif re.search(r'\b(bye|goodbye)\b', query_lower):
            return "Goodbye! Have a great day at the mall!"
        elif re.search(r'\b(how\s+are\s+you|what\'s\s+up)\b', query_lower):
            return "I'm doing great and ready to assist you! Feel free to ask me anything about the mall."
        elif re.search(r'\b(who\s+are\s+you|what\s+is\s+your\s+name)\b', query_lower):
            return "I'm SanHri-X, your smart mall assistant. I'm here to provide store information, timings, and directions."
    
    # Fallback response
    if language == "hi":
        return "मैं SanHri-X हूं। मैं आपकी मॉल से जुड़ी किसी भी जानकारी में मदद कर सकता हूं।"
    elif language == "mr":
        return "मी SanHri-X आहे. मी तुम्हाला मॉलशी संबंधित कोणत्याही माहितीत मदत करू शकतो."
    else:
        return "I'm SanHri-X, your mall assistant. I can help you with any mall-related information."

# --- Language-based prompt generator
def get_prompt_template(language: str) -> PromptTemplate:
    if language == "hi":
        template = """
आप SanHri-X हैं — एक बुद्धिमान और मित्रवत वर्चुअल असिस्टेंट, जो ग्राहकों की मॉल से संबंधित पूछताछ जैसे दिशा-निर्देश, स्टोर समय, और चालू ऑफ़र्स में मदद करता है।

नीचे दी गई जानकारी का उपयोग करके उपयोगकर्ता के प्रश्न का उत्तर दें। यदि कोई प्रासंगिक जानकारी नहीं मिलती है, तो विनम्रता से बताएं कि जानकारी उपलब्ध नहीं है।

⚠️ उत्तर सरल, बोलचाल की हिंदी भाषा में दें। प्रतीकों (*, -, आदि) का उपयोग न करें और सूचियों को वाक्य के रूप में प्रस्तुत करें।

संदर्भ:
{context}

प्रश्न:
{input}
"""
    elif language == "mr":
        template = """
तुम्ही SanHri-X आहात — एक हुशार आणि मैत्रीपूर्ण आभासी सहाय्यक, जो मॉलशी संबंधित विचारलेले प्रश्न जसे की दिशा, दुकाने उघडण्याचे वेळापत्रक आणि चालू ऑफर्स यास मदत करतो.

खाली दिलेल्या माहितीनुसार वापरकर्त्याच्या प्रश्नाचे उत्तर द्या. जर संबंधित माहिती सापडली नाही, तर नम्रपणे सांगा की माहिती उपलब्ध नाही.

⚠️ उत्तर सोप्या, बोलण्यात वापरल्या जाणाऱ्या मराठीत द्या. कृपया *, -, इत्यादी चिन्हांचा वापर करू नका. सूची वाक्यरूपात सांगा.

संदर्भ:
{context}

प्रश्न:
{input}
"""
    else:
        template = """
You are SanHri-X, a friendly and intelligent virtual assistant designed to help customers with mall-related queries like directions, store timings, and current offers.

Use the information below to answer the user's query. If no relevant data is found, say so politely.

⚠️ Format the answer in plain, spoken-friendly text. Avoid using symbols like '*', '-', or markdown. Present lists as sentences.

Context:
{context}

User Query:
{input}
"""
    return PromptTemplate.from_template(template)

# --- Get documents from DB with connection pooling
async def fetch_store_documents():
    try:
        # Use connection pooling if available
        if not database.is_connected:
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
                p.description AS offer,
                d.is_active,
                p.start_date,
                p.end_date
            FROM shops s
            LEFT JOIN store_operating_hours h 
                ON s.shop_id = h.shop_id
            LEFT JOIN store_promotions p 
                ON s.shop_id = p.shop_id
                AND CURRENT_DATE BETWEEN p.start_date AND p.end_date
            LEFT JOIN store_directory d 
                ON s.shop_id = d.store_id 
            ORDER BY s.shop_name
            LIMIT 100
        """)

        
        rows = await database.fetch_all(query)
        
        return [
            Document(
                page_content=(
                    f"{row['shop_name']} is located on floor {row['floor']}. "
                    f"It belongs to the '{row['category']}' category. "
                    f"On {row['day_of_week']}, it is "
                    f"{'closed' if row['is_closed'] else f'open from {row['open_time']} to {row['close_time']}'}."
                    f" Current offer: {row['offer'] or 'No current offers'}."
                ),
                metadata={"shop_name": row['shop_name'], "category": row['category']}
            )
            for row in rows
        ]
    except Exception as e:
        logging.error(f"Database fetch error: {e}")
        return []

# --- Get or create cached vector store
async def get_vectorstore():
    global _vectorstore_cache, _cache_timestamp, _documents_cache
    
    current_time = datetime.now()
    
    # Check if cache is valid
    if (_vectorstore_cache is not None and 
        _cache_timestamp is not None and 
        current_time - _cache_timestamp < CACHE_DURATION):
        return _vectorstore_cache
    
    try:
        # Fetch fresh documents
        docs = await fetch_store_documents()
        
        if not docs:
            logging.warning("No documents fetched from database")
            return None
            
        # Create vector store
        embeddings = get_embeddings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        final_docs = splitter.split_documents(docs)
        
        # Create FAISS index
        _vectorstore_cache = FAISS.from_documents(final_docs, embeddings)
        _cache_timestamp = current_time
        _documents_cache = final_docs
        
        logging.info(f"Vector store created with {len(final_docs)} documents")
        return _vectorstore_cache
        
    except Exception as e:
        logging.error(f"Vector store creation failed: {e}")
        return None

# --- Health check
@router.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "date": date.today().isoformat(),
        "cache_status": "active" if _vectorstore_cache else "empty",
        "cached_docs": len(_documents_cache)
    }

# --- Main /ask endpoint with caching and optimization
@router.post("/ask", response_model=QueryResponse)
async def ask_virtual_assistant(query_input: QueryInput):
    start_time = datetime.now()
    query = query_input.query.strip()
    language = query_input.language.strip().lower()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if language not in ["en", "hi", "mr"]:
        raise HTTPException(status_code=400, detail="Unsupported language. Use 'en', 'hi', or 'mr'.")

    try:
        # --- Quick response for general queries - NO DATABASE ACCESS
        if is_general_query(query, language):
            response_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"General query processed in {response_time:.2f}s")
            return {
                "answer": get_general_response(query, language),
                "context": []
            }

        # --- Get cached vector store
        vectorstore = await get_vectorstore()
        if not vectorstore:
            fallback_message = {
                "hi": "क्षमा करें, अभी मॉल की जानकारी उपलब्ध नहीं है। कृपया बाद में पुनः प्रयास करें।",
                "mr": "क्षमस्व, सध्या मॉलची माहिती उपलब्ध नाही. कृपया नंतर प्रयत्न करा.",
                "en": "Sorry, mall information is currently unavailable. Please try again later."
            }
            return {
                "answer": fallback_message.get(language, fallback_message["en"]),
                "context": []
            }

        #   Fast RAG processing with cached components
        model = get_model()
        prompt = get_prompt_template(language)
        
        # Create retriever with optimized settings
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 100}  # Limit to top 3 results
        )
        
        # Create processing chains
        document_chain = create_stuff_documents_chain(model, prompt)
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        # Process query with timeout
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: retriever_chain.invoke({"input": query})
            ),
            timeout=15.0  # 15 second timeout
        )

        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"RAG query processed in {response_time:.2f}s")

        return {
            "answer": result.get("answer") or result.get("output", 
                "माफ़ कीजिए, मुझे उत्तर नहीं मिला।" if language == "hi" 
                else "क्षमस्व, उत्तर सापडले नाही." if language == "mr" 
                else "Sorry, I couldn't find a response."
            ),
            "context": [doc.page_content for doc in result.get("context", [])]
        }

    except asyncio.TimeoutError:
        logging.error("Query processing timeout")
        timeout_message = {
            "hi": "क्षमा करें, प्रश्न का उत्तर देने में अधिक समय लग रहा है। कृपया पुनः प्रयास करें।",
            "mr": "क्षमस्व, प्रश्नाचे उत्तर देण्यासाठी जास्त वेळ लागत आहे. कृपया पुन्हा प्रयत्न करा.",
            "en": "Sorry, the query is taking too long to process. Please try again."
        }
        return {
            "answer": timeout_message.get(language, timeout_message["en"]),
            "context": []
        }
    except Exception as e:
        logging.exception("Error in RAG assistant:")
        error_message = {
            "hi": "क्षमा करें, तकनीकी समस्या के कारण उत्तर नहीं दे सका। कृपया पुनः प्रयास करें।",
            "mr": "क्षमस्व, तांत्रिक समस्येमुळे उत्तर देता आले नाही. कृपया पुन्हा प्रयत्न करा.",
            "en": "Sorry, I encountered a technical issue. Please try again."
        }
        return {
            "answer": error_message.get(language, error_message["en"]),
            "context": []
        }

# Cache management endpoints
@router.post("/refresh-cache")
async def refresh_cache():
    global _vectorstore_cache, _cache_timestamp
    try:
        _vectorstore_cache = None
        _cache_timestamp = None
        await get_vectorstore()  # Rebuild cache
        return {"status": "Cache refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache refresh failed: {str(e)}")

@router.get("/cache-stats")
async def cache_stats():
    return {
        "cache_active": _vectorstore_cache is not None,
        "cache_age_minutes": (datetime.now() - _cache_timestamp).total_seconds() / 60 if _cache_timestamp else None,
        "cached_documents": len(_documents_cache),
        "cache_expires_in_minutes": (CACHE_DURATION.total_seconds() / 60) - ((datetime.now() - _cache_timestamp).total_seconds() / 60) if _cache_timestamp else None
    }
