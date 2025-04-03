import os
import pandas as pd

# Define dataset folder (Make sure this path is correct)
base_path = "aclImdb"

# Read positive reviews
train_pos_files = os.listdir(os.path.join(base_path, "train/pos"))
train_pos = []
for file in train_pos_files:
    with open(os.path.join(base_path, "train/pos", file), "r", encoding="utf-8") as f:
        train_pos.append((f.read(), 1))  # 1 = Positive Sentiment

# Read negative reviews
train_neg_files = os.listdir(os.path.join(base_path, "train/neg"))
train_neg = []
for file in train_neg_files:
    with open(os.path.join(base_path, "train/neg", file), "r", encoding="utf-8") as f:
        train_neg.append((f.read(), 0))  # 0 = Negative Sentiment

# Combine positive & negative reviews
train_data = train_pos + train_neg

# Convert to DataFrame
df_train = pd.DataFrame(train_data, columns=["review", "sentiment"])

# Save as CSV
df_train.to_csv("imdb_reviews.csv", index=False)

print("✅ IMDB Dataset Saved as 'imdb_reviews.csv' Successfully!")
