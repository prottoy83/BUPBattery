import os
import pulp
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_KEY"))

@app.get("/health")
async def health_check():
    return {"status": "ok"}