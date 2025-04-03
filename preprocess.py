import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (if not already downloaded)
nltk.download("stopwords")

# Load the CSV file
df = pd.read_csv("imdb_reviews.csv")

# Define a function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags (if any)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    words = text.split()  # Tokenize by splitting on whitespace
    # Remove stopwords from the tokenized words
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Apply the cleaning function to each review
df["cleaned_review"] = df["review"].apply(clean_text)

# Save the cleaned data to a new CSV file
df.to_csv("imdb_reviews_cleaned.csv", index=False)

print("✅ Text Preprocessing Completed! Cleaned data saved as 'imdb_reviews_cleaned.csv'")
