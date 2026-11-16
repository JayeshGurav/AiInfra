import os
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

app = FastAPI(title="Enterprise RAG Chat")
INDEX = Path(__file__).parent / "index.html"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX)


@app.get("/config.js")
async def frontend_config() -> Response:
    api_url = os.getenv("FRONTEND_API_URL", "http://localhost:8000").rstrip("/")
    return Response(
        f"window.RAG_API_URL = {json.dumps(api_url)};",
        media_type="application/javascript",
    )
