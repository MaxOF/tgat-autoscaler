"""Compatibility entry point. The supported application is defined in main.py."""

from main import app, create_app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
