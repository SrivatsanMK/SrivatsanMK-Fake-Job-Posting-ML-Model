# 🕵️ Fake Job Posting Detection System  
### Machine Learning + Flask Web Application

A complete **Fake Job Posting Detection System** that uses **Machine Learning and Natural Language Processing (NLP)** to identify whether a job posting is **Real** or **Fake**.  
This project is designed for **academic use, portfolios, and real-world demonstration** of text classification.

> **Developed by:** Srivatsan Mk  

---

## 🚀 Project Overview

Fake job postings are increasingly common on online job portals. This project aims to **automatically detect fraudulent job postings** using a trained Machine Learning model.

The system analyzes job descriptions and related textual features to classify postings as:
- ✅ **Real Job**
- ❌ **Fake Job**

The model is deployed using a **Flask web application**, allowing users to enter job details and instantly receive predictions.

---

## 🎯 Key Features

- 🧠 **Machine Learning-based Text Classification**
- 📝 Uses **NLP techniques** for text preprocessing
- 🌐 **Flask Web Interface** for easy interaction
- ⚡ Real-time prediction
- 📦 Pre-trained model (`.pkl`)
- 🎨 Clean and simple UI
- 💼 Suitable for **Final Year Projects & Portfolios**

---

## 🧠 Machine Learning Details

### 🔍 Problem Type
- **Binary Classification**
  - Fake Job (1)
  - Real Job (0)

### 📊 Dataset
- Source: Fake Job Postings Dataset
- Contains:
  - Job title
  - Company profile
  - Job description
  - Requirements
  - Benefits
  - Employment type
  - Location

### 🛠 Techniques Used
- Text Cleaning
- Tokenization
- Stopword Removal
- TF-IDF Vectorization
- Supervised ML Classification

### 🤖 Model
- Trained using Scikit-learn
- Saved as a `.pkl` file for reuse

---

## 🏗️ Project Structure
```
Fake-Job-Posting-ML-Model/
│
├── app.py # Flask backend
├── Model.ipynb # Model training notebook
├── fake_job_model.pkl # Trained ML model
├── fake_job_postings.csv # Dataset
├── requirements.txt # Project dependencies
│
├── templates/
│ └── index.html # Frontend HTML
│
├── static/
│ └── style.css # Styling
│
└── README.md # Project documentation
```

---

## 🖥️ Web Application Flow
```
User Input → Flask Backend → ML Model (.pkl) → Prediction Result
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SrivatsanMK/SrivatsanMK-Fake-Job-Posting-ML-Model.git
cd SrivatsanMK-Fake-Job-Posting-ML-Model
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
python app.py
```

### 5️⃣ Open in Browser
```bash
http://127.0.0.1:5000/
```

---

### 📊 Output
- ✔ Real Job Posting
- ❌ Fake Job Posting

The prediction result is displayed clearly on the web interface.

---

### 📈 Use Cases
- Online job portals
- Recruitment platforms
- Fraud detection systems
- Academic ML projects
- Portfolio demonstration

---

### 🔮 Future Enhancements
- Add deep learning models (LSTM / BERT)
- Improve UI with modern frameworks
- Add confidence score for predictions
- Deploy on cloud (Render / AWS / Heroku)
- Add admin dashboard

---

### 🙌 Acknowledgements
- Scikit-learn
- Flask
- Pandas
- NumPy
- Open-source datasets
