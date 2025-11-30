"""
FastAPI Application for Model Serving - Simplified Version
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .model_loader import ModelLoader
from .inference import InferenceEngine
from ..utils.config import ServingConfig, load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID for recommendations")
    top_k: int = Field(10, description="Number of recommendations to return", le=50)
    model: str = Field("ensemble", description="Model to use: xgb, lstm, tcn, ensemble")

class ProductRecommendation(BaseModel):
    product_id: int
    score: float
    model: str

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[ProductRecommendation]
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str

# Global variables
app_start_time = datetime.now()

# Create FastAPI app
app = FastAPI(
    title="Instacart Recommendation API",
    description="Production API for product recommendations using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Instacart Recommendation API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        message=f"API running for {uptime:.1f} seconds"
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get product recommendations for a user"""
    
    try:
        # For demo purposes, return dummy recommendations
        dummy_recommendations = []
        for i in range(request.top_k):
            dummy_recommendations.append(ProductRecommendation(
                product_id=1000 + i,
                score=0.9 - (i * 0.05),
                model=request.model
            ))
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=dummy_recommendations,
            model_used=request.model,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Recommendation failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Setup logging
    setup_logging()
    
    logger.info("Instacart Recommendation API initialized")
    
    return app

def main():
    """Run the FastAPI server"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Instacart Recommendation API")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Create app with config
    app_instance = create_app(args.config)
    
    # Run server
    uvicorn.run(
        app_instance,
        host=args.host,
        port=args.port,
        reload=args.reload,
        access_log=True
    )

if __name__ == "__main__":
    main()