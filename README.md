# 🛒 Instacart Consumer Behavior Analysis - Production ML System

[![CI Pipeline](https://github.com/Rekk-tech/Instacart-consumer-behavior-analysic/workflows/CI%20Pipeline/badge.svg)](https://github.com/Rekk-tech/Instacart-consumer-behavior-analysic/actions)
[![Docker Build](https://github.com/Rekk-tech/Instacart-consumer-behavior-analysic/workflows/Docker%20Build%20&%20Deploy/badge.svg)](https://github.com/Rekk-tech/Instacart-consumer-behavior-analysic/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready machine learning system for Instacart customer behavior analysis and next purchase prediction**

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)  
- [🚀 Quick Start](#-quick-start)
- [🐳 Docker Deployment](#-docker-deployment)
- [⚙️ CI/CD Pipeline](#️-cicd-pipeline)
- [📊 Models & Performance](#-models--performance)
- [🔧 Development](#-development)
- [📖 API Documentation](#-api-documentation)

## 🎯 Overview

This project transforms exploratory Instacart analysis notebooks into a **production-grade ML system** with:

- 🤖 **Multiple ML Models**: XGBoost, LightGBM, LSTM, TCN
- 🔄 **End-to-End Pipeline**: Data ingestion → Feature engineering → Training → Serving
- 🚀 **REST API**: FastAPI-based model serving
- 📊 **Interactive Dashboard**: Streamlit-powered analytics
- 🐳 **Container Ready**: Docker & Docker Compose support
- ⚡ **CI/CD Pipeline**: Automated testing, linting, and deployment
- 📈 **Production Monitoring**: Logging, health checks, metrics

## 🏗️ Architecture

### Project Structure
```
├── 📁 src/                    # Core application code
│   ├── 📂 data/              # Data ingestion & ETL
│   ├── 📂 features/          # Feature engineering
│   ├── 📂 models/            # Model training & evaluation
│   ├── 📂 serving/           # API & model serving
│   ├── 📂 pipelines/         # Pipeline orchestration
│   └── 📂 utils/             # Utilities & configuration
├── 📁 configs/               # Model & training configurations
├── 📁 data/                  # Data storage (raw/processed/features)
├── 📁 models/                # Trained model artifacts
├── 📁 streamlit_app/         # Interactive dashboard
├── 📁 .github/workflows/     # CI/CD automation
├── 📁 docs/                  # Documentation & reports
└── 📄 docker-compose.yml     # Multi-container deployment
```

### System Components
- **Data Pipeline**: Automated ETL with feature engineering
- **Model Training**: Multiple algorithms with hyperparameter tuning  
- **API Server**: FastAPI with Pydantic validation
- **Web Dashboard**: Streamlit for business insights
- **Containerization**: Docker for consistent deployment
- **CI/CD**: GitHub Actions for automated testing & deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- Docker & Docker Compose (for containerized deployment)
- Git

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Rekk-tech/Instacart-consumer-behavior-analysic.git
cd Instacart-consumer-behavior-analysic

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### 2. Run System Test
```bash
python test_system.py
# Expected output: "🎯 System Status: READY FOR PRODUCTION!"
```

### 3. Training Models

```bash
# Train XGBoost (recommended for production)
python -m src.main train --config configs/train_xgb.yaml

# Train LSTM for sequential modeling
python -m src.main train --config configs/train_lstm.yaml

# Train TCN (Temporal Convolutional Network)  
python -m src.main train --config configs/train_tcn.yaml
```

### 4. Start Services

#### Option A: Individual Services
```bash
# Start API server
python -m src.main serve --config configs/train_xgb.yaml --port 8000

# Start dashboard (in separate terminal)
streamlit run streamlit_app/app.py
```

#### Option B: Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 5. Access Applications
- 🔗 **API Server**: http://localhost:8000
- 📊 **Dashboard**: http://localhost:8501  
- 📋 **API Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker build -t instacart-recommender .

# Run API server
docker run -p 8000:8000 instacart-recommender

# Run dashboard
docker run -p 8501:8501 instacart-recommender \
  streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0
```

### Multi-Container with Docker Compose
```bash
# Production deployment
docker-compose up -d

# Development with auto-reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Scale services
docker-compose up --scale api=3 --scale dashboard=2

# View service status
docker-compose ps
```

### Container Management
```bash
# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

## ⚙️ CI/CD Pipeline

### GitHub Actions Workflows

#### 1. **CI Pipeline** (`.github/workflows/ci.yml`)
Runs on every push/PR:
- ✅ **Code Linting**: flake8, black, isort
- ✅ **Unit Tests**: pytest with coverage
- ✅ **API Tests**: FastAPI endpoint testing  
- ✅ **Security Scan**: bandit, safety
- ✅ **Multi-Python**: Testing on 3.9, 3.10

#### 2. **Docker Build** (`.github/workflows/docker.yml`) 
Runs on main branch:
- 🐳 **Build Images**: Multi-arch Docker builds
- 📦 **Push Registry**: GitHub Container Registry
- 🚀 **Auto Deploy**: Staging environment
- ❤️ **Health Checks**: Automated validation

#### 3. **Release** (`.github/workflows/release.yml`)
Triggered by version tags:
- 🏷️ **Auto Tagging**: Semantic versioning
- 📋 **Release Notes**: Automated changelog
- 📦 **Artifacts**: Distribution packages

### Local Development Workflow
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black src/
isort src/  
flake8 src/

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run security checks
bandit -r src/
safety check
```

### Environment Variables
```bash
# .env file
PYTHON_ENV=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/archived/
```

## 📊 Models & Performance

### Available Models

| Model | Type | Use Case | Performance* |
|-------|------|----------|-------------|
| **XGBoost** | Tree-based | General recommendation | AUC: 0.85+ |
| **LightGBM** | Tree-based | Fast inference | AUC: 0.84+ |
| **LSTM** | Sequential | Time-series patterns | AUC: 0.79+ |
| **TCN** | Convolutional | Sequential modeling | AUC: 0.80+ |

*Performance on validation set

### Model Artifacts
```
models/archived/
├── xgboost_baseline_[timestamp].joblib
├── lightgbm_baseline_[timestamp].joblib  
├── lstm_final_[timestamp].h5
└── tcn_final_[timestamp].h5
```

## 🔧 Development

### Project Setup
```bash
# Clone and install
git clone [repo-url]
cd Instacart-consumer-behavior-analysic
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Adding New Models
1. Create trainer in `src/models/`
2. Add configuration in `configs/`  
3. Update `src/pipelines/train.py`
4. Add tests in `tests/`

### Code Standards
- **Formatting**: Black (line length 127)
- **Import sorting**: isort  
- **Linting**: flake8
- **Type hints**: mypy (optional)
- **Testing**: pytest + coverage

## 📖 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```
Response: `{"status": "healthy", "timestamp": "2025-11-30T..."}`

#### User Recommendations  
```http
POST /recommendations/
Content-Type: application/json

{
  "user_id": 12345,
  "num_recommendations": 10,
  "model_type": "xgboost"
}
```

#### Batch Predictions
```http
POST /batch-predict/
Content-Type: application/json

{
  "user_ids": [12345, 67890],
  "features": {...}
}
```

### Interactive API Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📈 Monitoring & Logging

### Application Logs
```bash
# View logs
tail -f logs/instacart_recommender_[timestamp].log

# Docker logs
docker-compose logs -f api
```

### Health Monitoring
- **API Health**: `/health` endpoint
- **Model Performance**: Logged in training
- **System Metrics**: Container stats

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`  
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Instacart Market Basket Analysis Dataset
- FastAPI and Streamlit communities
- Open source ML libraries

---

**📞 Support**: Open an issue for questions or bug reports  
**⭐ Star**: If this project helps you, please give it a star!

## 📚 Quick Reference

### Common Commands
```bash
# 🏃‍♂️ Quick Start
python test_system.py                    # System health check
python -m src.main train --config configs/train_xgb.yaml   # Train model
python -m src.main serve --config configs/train_xgb.yaml   # Start API
streamlit run streamlit_app/app.py       # Launch dashboard

# 🐳 Docker Commands  
docker-compose up -d                     # Start all services
docker-compose down                      # Stop services
docker-compose logs -f api               # View API logs
docker-compose ps                        # Service status

# 🔧 Development
pip install -e ".[dev]"                 # Dev installation
black src/ && isort src/                # Code formatting
pytest --cov=src                        # Run tests
flake8 src/                             # Linting

# 📦 Data Pipeline
python -m src.main pipeline --steps data etl --config configs/train_xgb.yaml

# 🚀 Production Deploy
docker build -t instacart-recommender .
docker run -p 8000:8000 instacart-recommender
```

### Environment Setup
```bash
# Python Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Docker Environment  
docker --version
docker-compose --version
```

### Troubleshooting
- **Port conflicts**: Change ports in `docker-compose.yml` or `.env`
- **Memory issues**: Reduce batch size in model configs
- **Model not found**: Check `models/archived/` directory
- **API errors**: Check logs in `logs/` directory
python -m src.main serve --config configs/train_xgb.yaml --port 8000 --reload
```

### 4. Chạy Pipeline riêng lẻ

```bash
# Chỉ chạy data processing
python -m src.main pipeline --steps data etl --config configs/train_xgb.yaml

# Chạy feature engineering
python -m src.main pipeline --steps data etl features --config configs/train_xgb.yaml
```

## 📊 API Endpoints

### Recommendation API
- `POST /recommend` - Get recommendations for single user
- `POST /recommend/batch` - Batch recommendations  
- `GET /models` - List loaded models
- `GET /health` - Health check

### Example API Usage

```python
import requests

# Single user recommendation
response = requests.post("http://localhost:8000/recommend", json={
    "user_id": 123,
    "top_k": 10,
    "model": "ensemble"
})

recommendations = response.json()["recommendations"]
```

## 🔧 Configuration

Configuration được quản lý qua YAML files trong thư mục `configs/`:

- `train_xgb.yaml` - XGBoost configuration
- `train_lstm.yaml` - LSTM configuration  
- `train_tcn.yaml` - TCN configuration

## 📈 Models Supported

### 1. XGBoost Baseline
- Tree-based model với user-item features
- Fast training và inference
- Explainable features importance

### 2. LSTM Sequential
- Sequential recommendation với user purchase history
- Embedding layer + LSTM layers
- Suitable cho temporal patterns

### 3. TCN (Temporal Convolutional Networks)
- Alternative sequential approach
- Dilated convolutions with residual connections
- Parallel processing advantage

### 4. Ensemble Methods
- Weighted combination của multiple models
- Stacking với meta-learner
- Hybrid tree + deep learning approaches

## 🎯 Pipeline Components

### Pipeline 1: ETL
```python
from src.data.ingest import DataIngester
from src.data.etl_rfm import RFMAnalyzer

# Load và preprocess raw data
ingester = DataIngester(config)
raw_data = ingester.load_all_data()

# RFM analysis
rfm_analyzer = RFMAnalyzer(config) 
rfm_features = rfm_analyzer.compute_rfm_features(orders)
```

### Pipeline 2: Feature Engineering
```python
from src.features import UserFeatureBuilder, ItemFeatureBuilder, SequenceBuilder

# User features
user_builder = UserFeatureBuilder(config)
user_features = user_builder.build_features(orders, products, aisles, departments)

# Sequence features cho deep learning
sequence_builder = SequenceBuilder(config)
sequences, targets, metadata = sequence_builder.build_sequences(orders, order_products)
```

### Pipeline 3: Model Training
```python
from src.models import XGBTrainer, LSTMTrainer, TCNTrainer

# Train XGBoost
xgb_trainer = XGBTrainer(config)
results = xgb_trainer.train(features, targets)

# Train LSTM
lstm_trainer = LSTMTrainer(config)
results = lstm_trainer.train(sequences, targets, vocab_size)
```

### Pipeline 4: Model Serving
```python
from src.serving import ModelLoader, InferenceEngine

# Load models
model_loader = ModelLoader(config)
model_loader.load_xgb_model("models/xgboost_model.joblib")

# Generate recommendations
inference_engine = InferenceEngine(config, model_loader)
recommendations = inference_engine.predict_ensemble(user_id=123, top_k=10)
```

## 🔍 Monitoring & Evaluation

### Metrics được track:
- **Classification**: AUC, Precision, Recall, F1-score
- **Ranking**: NDCG, MAP, MRR
- **Recommendation**: Precision@K, Recall@K, Hit Rate@K  
- **Business**: Coverage, Diversity, Novelty

### Model Performance Tracking
- Training history được save trong CSV format
- Model metadata và config được lưu cùng model artifacts
- API metrics: latency, throughput, error rates

## 🛠️ Development

### Code Structure
- **Modular design**: Mỗi component có thể test và develop độc lập
- **Configuration-driven**: Tất cả parameters qua YAML configs
- **Type hints**: Full typing support với mypy
- **Logging**: Structured logging với configurable levels

### Testing
```bash
# Run unit tests
pytest tests/

# Run specific test module  
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 🐳 Deployment

### Docker Support (TBD)
```dockerfile
FROM python:3.8-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

CMD ["python", "-m", "src.main", "serve", "--config", "configs/train_xgb.yaml"]
```

### Production Checklist
- [ ] Model versioning
- [ ] A/B testing framework  
- [ ] Model monitoring & alerting
- [ ] Data drift detection
- [ ] Automated retraining
- [ ] Load balancing
- [ ] Caching layer (Redis)
- [ ] Database integration

## 📝 Next Steps

1. **Database Integration**: Connect với PostgreSQL/MongoDB cho real-time data
2. **Caching Layer**: Redis cho user features và recommendations  
3. **Model Versioning**: MLflow integration cho experiment tracking
4. **A/B Testing**: Framework để test model performance
5. **Monitoring**: Prometheus + Grafana cho production monitoring
6. **CI/CD Pipeline**: Automated testing và deployment
7. **Feature Store**: Central feature repository
8. **Real-time Training**: Online learning capabilities

## 📚 Documentation

Detailed documentation cho từng component:
- [Data Pipeline Documentation](docs/data_pipeline.md)
- [Feature Engineering Guide](docs/feature_engineering.md)  
- [Model Training Guide](docs/model_training.md)
- [API Documentation](docs/api_documentation.md)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests cho new functionality
4. Ensure code quality checks pass
5. Submit pull request

