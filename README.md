# Email Spam Classifier (ML-Powered)

A full-stack web application that uses Machine Learning to classify emails as **Spam** or **Ham** (Legitimate). This project compares multiple classification models and provides a real-time confidence score for each prediction.

## Features
- **ML Engine:** Built with `scikit-learn`, utilizing text analysis.
- **Precision Focused:** Optimized using the **Precision metric** to minimize false positives (marking important mail as spam).
- **Probability Scoring:** Provides a confidence percentage using the `predict_proba` method.
- **Modern Stack:** React frontend for a clean UI and Node.js/Express backend for API handling.

## Tech Stack
- **Frontend:** React
- **Backend:** Node.js, Express
- **Machine Learning:** Python, Scikit-Learn, Pandas, Joblib
- **Environment:** Virtual Environments (venv) for dependency isolation

---

## Project Structure
```
email-spam-clf/
├── backend/            # Node.js Express server
├── frontend/           # React (Vite) application
├── model/              # Python scripts and trained .joblib models
├── .venv/              # Python virtual environment
└── requirements.txt    # Python dependencies
```
## Setup & Installation

## 1. Clone The repository

git clone https://github.com/Malav023/email-spam-classifier.git
cd email-spam-classifier

## 2. Machine Learning Setup (Python)

### Activate your virtual environment
 Windows:
      .\.venv\Scripts\activate
 Mac/Linux:
      source .venv/bin/activate

### Install dependencies
pip install -r requirements.txt

## 3. Node js 
cd backend
npm install
node server.js

## 4. React frontend
cd ../frontend
npm install
npm run dev

## 5. Python backend too 
cd ../backend 
uvicorn app:app --port 5001 --reload

## NOTE : run steps 3-5 in separated cmd's

## Model Evaluation
During development, various models (Naive Bayes, SVM, SGD) were compared. The final implementation prioritizes Precision to ensure that legitimate communications are never accidentally filtered out, maintaining high reliability for the user.

