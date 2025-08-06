from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from app.api import rag

from fastapi.middleware.cors import CORSMiddleware
from app.db import database

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
        "https://ai-mall.netlify.app"    # production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(rag.router)


# ---- DB lifecycle ----
@app.on_event("startup")
async def startup():
    # connect DB once at startup (creates connection pool)
    try:
        await database.connect()
        print("Database connected")
    except Exception as e:
        # log and optionally re-raise to fail the startup
        print("Failed to connect to database on startup:", e)
        raise

@app.on_event("shutdown")
async def shutdown():
    try:
        await database.disconnect()
        print("Database disconnected")
    except Exception as e:
        print("Error disconnecting database:", e)

# ---- run locally convenience ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)