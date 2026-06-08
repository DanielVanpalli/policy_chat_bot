from fastapi import Request
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from config import settings


async def auth_middleware(request: Request, call_next):
    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Missing or malformed token"})

    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except JWTError as e:
        return JSONResponse(status_code=401, content={"detail": f"Invalid token: {e}"})

    request.state.user_id = payload["sub"]
    request.state.user_payload = payload
    return await call_next(request)
