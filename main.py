from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class CodeRequest(BaseModel):
    code: str
    language: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return {"status": "ERROR", "reason": "GROQ_API_KEY missing from .env"}
    return {"status": "OK", "key_preview": key[:10] + "..."}

@app.post("/review")
def review_code(request: CodeRequest):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in .env")

    try:
        prompt = f"""You are an expert code reviewer. Review the following {request.language} code.

Provide your review in this exact format:

## Overall Score
Give a score out of 10 with one sentence explanation.

## What is Good
List 2-3 things done well.

## Bugs and Issues
List any bugs, errors, or problems found.

## Improvements
List 2-3 specific suggestions to improve the code.

## Corrected Code
Provide the improved version of the code.

Here is the code to review:
{request.code}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert code reviewer. Be detailed, helpful and constructive."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )

        return {"review": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")