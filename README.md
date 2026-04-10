# 🏥 Disease Prediction & Drug Recommendation System
### Final Year Major Project (2024)
### Department of Artificial Intelligence and Data Science (AI & DS)

A machine learning system that predicts diseases based on symptoms and recommends medications using patient review data analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.18+-red.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-orange.svg)

## 📋 Project Overview
This major project combines disease prediction and drug recommendation using machine learning and NLP techniques. The system analyzes patient drug reviews to provide data-driven insights for healthcare decisions.

Based on research paper: "An Intelligent Disease Prediction and Drug Recommendation Prototype by Using Multiple Approaches of Machine Learning Algorithms" by Suvendu Kumar Nayak et al.

## 🌟 Features
- ML-powered disease prediction from symptoms
- Drug recommendation based on patient reviews
- Sentiment analysis of drug reviews
- Interactive web interface using Streamlit
- Data visualization of predictions
- Multi-condition classification system

## 📊 Dataset Implementation
Uses UCI ML Drug Review dataset ([KUC Hackathon Winter 2018](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018))

### Dataset Features
- 215,063 patient drug reviews
- Patient condition information
- 10-star rating system
- Timestamps and usefulness ratings
- Detailed patient experiences

## 🔧 Technical Architecture

### Technologies Used
- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Pipeline**: Scikit-learn
- **NLP**: NLTK, BeautifulSoup4
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib

### Model Implementation
- Text preprocessing with NLTK
- TF-IDF vectorization with n-grams
- Passive Aggressive Classifier
- Naive Bayes implementation
- Drug recommendation engine

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
git clone https://github.com/darkxdd/Disease-Prediction-and-Drug-Recommendation-Prototype.git
cd Disease-Prediction-and-Drug-Recommendation-Prototype
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Results & Performance

### Model Accuracy
- Implemented multiple ML approaches:
  - Bag of Words (BoW)
  - TF-IDF with n-grams
  - Passive Aggressive Classifier
  - Naive Bayes Classifier

### Conditions Covered
- Birth Control
- Depression
- Diabetes Type 2
- High Blood Pressure
- Migraine
- Pneumonia
- Asthma (acute)
- Urinary Tract Infection
- ADHD
- Acne
... etc

## 📁 Project Structure

```
Disease-Prediction-and-Drug-Recommendation-Prototype/
├── app.py              # Main application file
├── model/             
│   ├── passmodel.pkl   # Trained ML model
│   └── tfidfvectorizer.pkl  # Text vectorizer
├── data/
│   └── custom_dataset.csv   # Drug reviews dataset
└── requirements.txt    # Project dependencies
```

## 🎯 How to Use

1. Select symptoms from the predefined list in the web interface
2. Click the "Predict" button
3. View the predicted condition and recommended drugs
4. Explore the interactive visualization of drug recommendations
## ⚠️ Disclaimer
This system is a prototype for educational purposes. Always consult healthcare professionals for medical advice.

## 👨‍💻 Project Team
- Student Name: Puneeth Raj
- Roll Number: 21831A7251
- Supervisor: Sheeltal Kundra
- Department: Artificial Intelligence and Data Science (AI & DS)

## 📧 Contact
Student Email: [puneeth@duck.com](mailto:puneeth@duck.com)

Project Link: [GitHub Repository](https://github.com/darkxdd/Disease-Prediction-and-Drug-Recommendation-Prototype)
