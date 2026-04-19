import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import upload, pipeline, results
from memory.memory_routes import router as memory_router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RoboData Pipeline API")

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_origins="*" is safe for local hackathon development.
# Tighten this to ["http://localhost:5173"] for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← accept all origins so CORS never blocks you
    allow_credentials=False,    # must be False when allow_origins="*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],   # needed for file download filename
)

# ── Global error handler — always returns CORS headers even on 500 ─────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url}:\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(pipeline.router)
app.include_router(results.router)
app.include_router(memory_router)       # ← Learning Memory System

@app.get("/health")
def health_check():
    return {"status": "ok"}

