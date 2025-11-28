from fastapi import FastAPI
from app.controller.tgat_controller import router as tgat_router
import uvicorn

def create_app() -> FastAPI:
    app = FastAPI(title="TGAT-Autoscaler", version="1.0.0")
    app.include_router(tgat_router)
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)