from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class VibeEchoRequest(BaseModel):
    message: string

@router.post("/echo")
async def echo(request: VibeEchoRequest):
    return {"message": f"Echo: {request.message}"}
