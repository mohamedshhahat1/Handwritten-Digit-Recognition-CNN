"""
Script to start the FastAPI prediction server.

Usage:
    python -m api.run_server
    # or
    python api/run_server.py
"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
