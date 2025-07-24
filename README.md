# AI Ethics Toolkit Prototype

A functional AI Ethics Toolkit prototype with FastAPI backend and Streamlit frontend for auditing AI models for bias, explainability, and hallucination detection.

## Features

- **Bias Detection**: Analyze models for demographic bias with intersectional analysis
- **Model Explainability**: Understand model decisions using SHAP and LIME
- **Hallucination Detection**: Identify false information in text generation
- **File Upload/Download**: Process models and datasets with in-memory storage
- **Real-time Progress Tracking**: Monitor analysis progress with background tasks
- **Session Persistence**: Maintain state using Python class variables

## Architecture

- **Backend**: FastAPI with in-memory storage using Python dictionaries
- **Frontend**: Streamlit UI with direct integration to core analysis modules
- **State Management**: Session-based without external dependencies
- **Processing**: Background tasks for compute-intensive operations

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: `fastapi`, `uvicorn`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `shap`, `lime`, `plotly`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-ethics-toolkit.git
   cd ai-ethics-toolkit
   ```

2. Install dependencies:
   ```bash
   pip install fastapi uvicorn streamlit pandas numpy scikit-learn joblib shap lime plotly pydantic-settings
   ```

### Running the Application

1. Start the backend server:
   ```bash
   python run_backend.py
   ```

2. Start the frontend server:
   ```bash
   python run_frontend.py
   ```

3. Access the application:
   - Frontend: http://localhost:12002
   - Backend API: http://localhost:12000
   - API Documentation: http://localhost:12000/docs

## Usage

1. **Upload Model**: Upload a trained model (.pkl or .joblib format)
2. **Upload Dataset**: Upload a dataset (.csv format)
3. **Run Analysis**: Select analysis type and configure parameters
4. **View Results**: Explore visualizations and recommendations

## API Endpoints

### Bias Detection

- `POST /api/v1/bias/upload/model`: Upload a model file
- `POST /api/v1/bias/upload/dataset`: Upload a dataset file
- `POST /api/v1/bias/audit`: Run a comprehensive bias audit
- `GET /api/v1/bias/task/{task_id}`: Get task status
- `GET /api/v1/bias/results/{session_id}`: Get bias audit results
- `POST /api/v1/bias/quick-check`: Run a quick bias check

### Explainability

- `POST /api/v1/explainability/explain`: Run explainability analysis
- `POST /api/v1/explainability/feature-importance`: Get feature importance

### Hallucination Detection

- `POST /api/v1/hallucination/detect`: Detect hallucinations in text

## Project Structure

```
ai-ethics-toolkit/
├── backend/
│   ├── api/
│   │   ├── middleware/
│   │   ├── models/
│   │   └── routes/
│   ├── services/
│   └── main.py
├── core/
│   ├── bias/
│   ├── explainability/
│   ├── hallucination/
│   ├── privacy/
│   └── shared/
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── utils/
│   └── app.py
├── logs/
├── run_backend.py
└── run_frontend.py
```

## Development Notes

- This is a prototype implementation focused on functionality
- Uses in-memory storage instead of a database for simplicity
- No authentication system implemented in this version
- Designed for single-machine deployment for demonstration purposes

## License

This project is licensed under the MIT License - see the LICENSE file for details.