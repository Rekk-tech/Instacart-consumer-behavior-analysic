#!/usr/bin/env python3
"""
Simple test API server for debugging
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/recommend")
async def get_recommendations(request: dict):
    # Simple mock response
    return {
        "user_id": request.get("user_id", 1),
        "recommendations": [
            {"rank": 1, "product_id": 1001, "product_name": "Test Product 1", "score": 0.85},
            {"rank": 2, "product_id": 1002, "product_name": "Test Product 2", "score": 0.78}
        ],
        "model_used": request.get("model", "test")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)