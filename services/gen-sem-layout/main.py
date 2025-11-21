from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class LayoutRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_layout(request: LayoutRequest):
    return {"status": "stub", "layout": "placeholder"}
