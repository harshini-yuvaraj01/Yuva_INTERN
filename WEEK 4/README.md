
# WEEK 4 - FINAL ML CAPSTONE

## Breast Cancer Classification Using Machine Learning

### Project Overview

This project implements an end-to-end Machine Learning
application using Python, Scikit-learn, Joblib and FastAPI.

The system predicts Benign or Malignant classifications
using the Breast Cancer Wisconsin dataset.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- HTML
- CSS
- JavaScript

### Installation

Install the required dependencies:

pip install -r requirements.txt

### Train the Model

Run:

python train.py

The training program performs:

1. Dataset loading
2. Data preprocessing
3. Train-test splitting
4. Pipeline creation
5. Model training
6. Model evaluation
7. Model serialization

The following files are generated:

- model.pkl
- metrics.json

### Start FastAPI

Run:

uvicorn app:app --reload

### Application URLs

Frontend:
http://127.0.0.1:8000/

Swagger UI:
http://127.0.0.1:8000/docs

Health Check:
http://127.0.0.1:8000/health

Feature Information:
http://127.0.0.1:8000/features

### API Endpoints

GET / - Displays frontend.

GET /health - Checks API status.

GET /features - Returns feature names and defaults.

POST /predict - Generates predictions.

### Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Deployment Workflow

The model is trained using Scikit-learn.
Joblib serializes the trained pipeline.
FastAPI loads the model and exposes REST endpoints.
The HTML frontend communicates with FastAPI using
JavaScript Fetch API.

### Conclusion

This project demonstrates the complete Machine Learning
workflow, including training, evaluation, serialization
and deployment.

### Disclaimer

This project is an educational prototype and is not
intended for medical diagnosis or treatment.
