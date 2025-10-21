from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient

app = FastAPI()
client = InferenceClient(model="openai/gpt-oss-20b")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    messages = [{"role": "system", "content": "You are a friendly AI chatbot."}]

    response = client.chat_completion(messages, max_tokens=256)
    answer = response.choises[0].message["content"]

    return JSONResponse({"reply": answer})