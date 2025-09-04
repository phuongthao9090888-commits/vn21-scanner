from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bot VN21 Scanner đang chạy trên Render!"}
