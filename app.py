from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Health check endpoint: hỗ trợ cả GET và HEAD
@app.get("/healthz")
@app.head("/healthz")
def healthz():
    return {"status": "ok"}

# ==== Các phần logic chính của bot ====
# (scanner, breakout detection, volume filter, Darvas, CANSLIM, Zanger, Telegram push…)
# Bạn giữ nguyên code phân tích của mình ở dưới đây
# chỉ cần chắc chắn KHÔNG xoá đoạn healthz phía trên.

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
