# Dockerfile for Instacart Recommendation System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/ ./data/
COPY streamlit_app/ ./streamlit_app/

# Create logs directory
RUN mkdir -p logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "src.main", "serve", "--config", "configs/train_xgb.yaml", "--port", "8000"]