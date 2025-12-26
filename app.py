from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

import uvicorn
from src.models import GCSRequest, ResponseModel
from src.services.vsr_service import VSRService

app = FastAPI(title="VSR API", description="API for video speech recognition")
vsr_service = VSRService()

# W3C Trace Context propagation
from opentelemetry.propagate import extract

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "healthy"})

@app.post("/predictions/vsr", response_model=ResponseModel)
async def process_video(data: GCSRequest, request: Request):
    """
    Process video from Google Cloud Storage
    """
    parent_context = extract(request.headers)
    with vsr_service.tracer.start_as_current_span(
        "vsr_request", context=parent_context
    ):
        return await vsr_service.process_video(data)

@app.post("/upload", response_model=ResponseModel)
async def upload_video(file: UploadFile = File(...), request: Request = None):
    """
    Process uploaded video file
    """
    parent_context = extract(request.headers) if request else None
    with vsr_service.tracer.start_as_current_span(
        "vsr_request", context=parent_context
    ):
        return await vsr_service.process_video(file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
