"""
FastAPI Application for Model Serving
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
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
    exclude_purchased: bool = Field(True, description="Exclude already purchased products")

class BatchRecommendationRequest(BaseModel):
    user_ids: List[int] = Field(..., description="List of user IDs")
    top_k: int = Field(10, description="Number of recommendations per user", le=50)
    model: str = Field("ensemble", description="Model to use")

class ProductRecommendation(BaseModel):
    product_id: int
    score: float
    model: str
    product_name: Optional[str] = None
    contributing_models: Optional[List[str]] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[ProductRecommendation]
    model_used: str
    timestamp: str
    processing_time_ms: float

class BatchRecommendationResponse(BaseModel):
    results: Dict[int, List[ProductRecommendation]]
    model_used: str
    timestamp: str
    total_processing_time_ms: float

class ModelStatus(BaseModel):
    model_name: str
    type: str
    status: str
    loaded_at: str
    path: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: List[str]
    uptime_seconds: float

# Global variables
config = None
model_loader = None
inference_engine = None
app_start_time = datetime.now()

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application"""
    
    global app, config, model_loader, inference_engine
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        # Use default serving config
        config = ServingConfig()
    
    # Setup logging
    setup_logging()
    
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
    
    # Initialize components
    global model_loader, inference_engine
    model_loader = ModelLoader(config.serving)
    inference_engine = InferenceEngine(config.serving, model_loader)
    
    # Add event handlers
    @app.on_event("startup")
    async def startup_event():
        """Initialize the application on startup"""
        
        logger.info("Starting Instacart Recommendation API...")
        
        # Load models
        model_path = Path(config.serving.model_path)
        
        # Try to load available models
        models_loaded = []
        
        if model_loader.load_xgb_model(model_path):
            models_loaded.append("xgb")
        
        if model_loader.load_lstm_model(model_path):
            models_loaded.append("lstm")
            
        if model_loader.load_tcn_model(model_path):
            models_loaded.append("tcn")
        
        # Load static data
        data_path = Path("data/processed")
        if data_path.exists():
            inference_engine.load_static_data(data_path)
        
        logger.info(f"API started with models: {models_loaded}")
    
    return app

# Create app instance for route definitions
app = create_app()

@app.get("/health", response_model=HealthResponse)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=model_loader.list_loaded_models(),
        uptime_seconds=uptime
    )

@app.get("/models", response_model=List[ModelStatus])
async def get_models():
    """Get information about loaded models"""
    
    models_info = []
    
    for model_name in model_loader.list_loaded_models():
        info = model_loader.get_model_info(model_name)
        models_info.append(ModelStatus(
            model_name=model_name,
            type=info['type'],
            status="loaded",
            loaded_at=info['loaded_at'],
            path=info['path']
        ))
    
    return models_info

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get product recommendations for a user"""
    
    start_time = datetime.now()
    
    try:
        # Generate recommendations
        if request.model == "ensemble":
            recommendations = inference_engine.predict_ensemble(
                user_id=request.user_id,
                top_k=request.top_k
            )
        elif request.model == "xgb":
            recommendations = inference_engine.predict_xgb(
                user_id=request.user_id,
                top_k=request.top_k
            )
        elif request.model == "lstm":
            recommendations = inference_engine.predict_lstm(
                user_id=request.user_id,
                top_k=request.top_k
            )
        elif request.model == "tcn":
            recommendations = inference_engine.predict_tcn(
                user_id=request.user_id,
                top_k=request.top_k
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}. Available: xgb, lstm, tcn, ensemble"
            )
        
        # Convert to response format
        rec_objects = []
        for rec in recommendations:
            rec_obj = ProductRecommendation(**rec)
            rec_objects.append(rec_obj)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=rec_objects,
            model_used=request.model,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Recommendation failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/recommend/batch", response_model=BatchRecommendationResponse)
async def get_batch_recommendations(request: BatchRecommendationRequest):
    """Get recommendations for multiple users"""
    
    start_time = datetime.now()
    
    try:
        if len(request.user_ids) > config.serving.batch_inference_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.user_ids)} exceeds limit {config.serving.batch_inference_size}"
            )
        
        # Generate batch recommendations
        batch_results = inference_engine.get_batch_recommendations(
            user_ids=request.user_ids,
            model_name=request.model,
            top_k=request.top_k
        )
        
        # Convert to response format
        formatted_results = {}
        for user_id, recommendations in batch_results.items():
            rec_objects = []
            for rec in recommendations:
                rec_obj = ProductRecommendation(**rec)
                rec_objects.append(rec_obj)
            formatted_results[user_id] = rec_objects
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchRecommendationResponse(
            results=formatted_results,
            model_used=request.model,
            timestamp=datetime.now().isoformat(),
            total_processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch recommendation failed: {str(e)}")

@app.post("/models/{model_name}/reload")
async def reload_model(model_name: str, background_tasks: BackgroundTasks):
    """Reload a specific model"""
    
    def reload_task():
        model_path = Path(config.serving.model_path)
        success = model_loader.reload_model(model_name, model_path)
        if success:
            logger.info(f"Model {model_name} reloaded successfully")
        else:
            logger.error(f"Failed to reload model {model_name}")
    
    background_tasks.add_task(reload_task)
    
    return {"message": f"Model {model_name} reload initiated"}

@app.post("/cache/clear")
async def clear_cache():
    """Clear inference caches"""
    
    inference_engine.clear_cache()
    
    return {"message": "Cache cleared successfully"}

@app.get("/")
async def root():
    """Root endpoint"""
    
    return {
        "message": "Instacart Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "models_available": model_loader.list_loaded_models()
    }

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
    app = create_app(args.config)
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        access_log=True
    )

if __name__ == "__main__":
    main()