Sentiment Analysis NLP Project – Explanation
This project is about analyzing IMDB movie reviews and classifying them as positive or negative using Natural Language Processing (NLP) and Machine Learning. 
Below is a step-by-step explanation of the entire process:

📌 Projects
Sentiment Analysis on IMDB Reviews
Tools & Technologies: Python, NLTK, Scikit-learn, Pandas, Streamlit, Joblib
Developed a Sentiment Analysis model to classify IMDB movie reviews as Positive or Negative.
Collected and preprocessed 25,000+ reviews from the IMDB dataset, removing stop words and applying tokenization.
Extracted key features using TF-IDF Vectorization and trained a Naïve Bayes Classifier to predict sentiments.
Achieved high accuracy in sentiment classification by fine-tuning hyperparameters.
Deployed the model using Streamlit, enabling users to enter text and get real-time sentiment predictions.


# 🎬 Sentiment Analysis on IMDB Reviews

This is a **Natural Language Processing (NLP)** project built with **Python, LSTM, and Streamlit UI** to classify movie reviews as **positive or negative**. Users can input their own reviews to get real-time sentiment predictions using a trained deep learning model.

---

## 🔍 Overview

- **Model Used:** LSTM (Long Short-Term Memory)
- **Data Source:** IMDB Movie Reviews Dataset
- **Frontend:** Streamlit
- **Backend:** TensorFlow/Keras
- **Tokenizer:** Fitted and saved using Keras Tokenizer

---

## 📌 Features

- Realtime sentiment prediction from user text
- NLP pipeline: cleaning, tokenization, padding
- Simple and interactive Streamlit UI
- Deployed-ready Python app

---

## 🧠 Tech Stack

| Layer        | Tools / Libraries                     |
|--------------|----------------------------------------|
| Language     | Python                                 |
| NLP          | NLTK, Tokenizer, Word Embeddings       |
| ML Model     | TensorFlow, Keras (LSTM)               |
| Visualization| Streamlit                              |

---

## 🚀 Installation
pip install -r requirements.txt


#Run the app
streamlit run main.py




✅ Final Output: Interactive Web App where users enter a review, and the app predicts sentiment.


