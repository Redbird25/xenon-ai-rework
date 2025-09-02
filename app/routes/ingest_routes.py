from fastapi import APIRouter, BackgroundTasks
from app.schemas import IngestRequest, IngestResponse
from app.ingest_job import run_ingest_job
import uuid

router = APIRouter(prefix="/ai/ingest", tags=["ingest"])

@router.post("/resources", response_model=IngestResponse)
async def ingest_resources(req: IngestRequest, background: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background.add_task(run_ingest_job, req, job_id)
    print(f"{IngestResponse}.")
    return IngestResponse(status="accepted", job_id=job_id)
