from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api import rag

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SANHRI-X Backend",
    description="API documentation for the SANHRI-X Mall AI Assistant backend, including Admin Panel endpoints.",
    version="1.0.0",
    contact={
        "name": "TechnoNexis Backend Team",
        "email": "backend@technonexis.com"
    },
    license_info={
        "name": "MIT License",
    }
)


# ✅ Add CORS middleware (IMPORTANT: Add early)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # local frontend
        # "https://admin.sanhri.com"    # production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(rag.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)