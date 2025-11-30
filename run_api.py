#!/usr/bin/env python3
"""
Simple script to run the API server
"""

import uvicorn
from src.serving.api import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)