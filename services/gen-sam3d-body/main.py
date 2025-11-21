from fastapi import FastAPI

app = FastAPI()

@app.post("/reconstruct")
async def reconstruct_body():
    return {"status": "stub", "body": "placeholder"}
