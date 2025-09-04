# ---- Health endpoints (accept GET & HEAD) ----
from fastapi import Response

@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    # Trả về 200 cho cả GET/HEAD, không body với HEAD theo chuẩn HTTP
    return Response(content="OK", media_type="text/plain")

# (tuỳ chọn) root cũng chấp nhận HEAD để ping thoải mái
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return Response(content="alive", media_type="text/plain")
