from fastapi import FastAPI

app = FastAPI()

@app.post("/reconstruct")
async def reconstruct_object():
    return {"status": "stub", "object": "placeholder"}
