from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app import build_qa
from typing import Any
import tempfile
import traceback
import os

app = FastAPI()
qa_holder: dict[str, Any] = {"qa": None}

#allow front-end requests
app.middleware("http")(CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
))
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print("📄 Uploaded PDF path:", tmp_path)
        qa_holder["qa"] = build_qa(tmp_path)
        print("🧠 QA object created:", qa_holder["qa"])
        return JSONResponse({"message": "✅ PDF uploaded and processed!"})
    except Exception as e:
        print("❌ ERROR during upload_pdf:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)})




@app.post("/ask")
async def ask(request: str = Form(...)):
    try:
        if qa_holder["qa"] is None:
            return JSONResponse({"reply": "⚠️ Please upload a PDF first."})
        print("🔹 Asking question:", request)
        result = qa_holder["qa"].invoke(request)
        print("✅ Invoke result:", result)
        return JSONResponse({"reply": result["result"]})
    except Exception as e:
        print("❌ ERROR during ask:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)})

@app.get("/health")
def health():
    return {"status": "ok"}