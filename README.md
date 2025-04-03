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


Example structure:
VS-code
my_project/
├── aclImdb/
│   ├── train/
│   │   ├── pos/
│   │   ├── neg/
│   ├── test/
│   │   ├── pos/
│   │   ├── neg/
├── 
├──load_data.py (load acllmdb files)
├──save_csv.py (convert csv file)
├──preprocess.py(make new csv file)
	
Jupyter-Notebook:
	train_test_evaluate.ipynb-(save the trained model and the vectorizer)

VS-code:
├──	main.py(Terminal-streamlit run main.py)




1️⃣ Data Collection

The dataset used is the IMDB dataset (aclImdb), which contains 25,000 positive and 25,000 negative reviews.
These reviews are stored in text files inside two folders:
aclImdb/train/pos → Positive reviews
aclImdb/train/neg → Negative reviews



2️⃣ Data Preprocessing (Cleaning the text)

Before training the model, we need to clean and prepare the text data.
Steps included:
✔ Removing special characters (like @, #, !, etc.)
✔ Converting text to lowercase
✔ Removing stopwords (like "is", "the", "and")
✔ Applying tokenization (splitting text into words)
✔ Lemmatization (reducing words to their root form, e.g., "running" → "run")

✅ Output: A cleaned dataset (imdb_reviews_cleaned.csv)




3️⃣ Feature Extraction (Converting text to numerical data)

Since Machine Learning models work with numbers, we used TF-IDF Vectorization to convert the text into a numerical format.
TF-IDF (Term Frequency - Inverse Document Frequency) assigns importance to words.
Example: Common words like "movie" appear in many reviews, so they get lower importance.

✅ Output: A feature matrix (X_train, X_test), ready for model training.



4️⃣ Model Building -(jupyter notebook)

Used Naïve Bayes Classifier (best for text classification).
Trained the model on cleaned IMDB review data.
Achieved good accuracy in predicting sentiment.

✅ Output: Saved model as sentiment_model.pkl and vectorizer.pkl.




5️⃣ Building a Streamlit Web App -VS-code

Created a web interface where users can input a movie review.
The trained model predicts whether the review is positive or negative.
Used Streamlit to build the UI with a simple text box and a "Predict Sentiment" button.

✅ Final Output: Interactive Web App where users enter a review, and the app predicts sentiment.


