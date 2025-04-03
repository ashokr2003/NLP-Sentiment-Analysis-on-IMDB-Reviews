import os
import pandas as pd

# Define path to dataset (Make sure this is correct)
base_path = "aclImdb"  # The folder inside your VS Code project

# Read text data from 'train' folder
train_pos_files = os.listdir(os.path.join(base_path, "train/pos"))
train_neg_files = os.listdir(os.path.join(base_path, "train/neg"))

# Read positive reviews
train_pos = []
for file in train_pos_files:
    with open(os.path.join(base_path, "train/pos", file), "r", encoding="utf-8") as f:
        train_pos.append((f.read(), 1))  # 1 = Positive Sentiment

# Read negative reviews
train_neg = []
for file in train_neg_files:
    with open(os.path.join(base_path, "train/neg", file), "r", encoding="utf-8") as f:
        train_neg.append((f.read(), 0))  # 0 = Negative Sentiment

# Combine positive & negative reviews
train_data = train_pos + train_neg

# Convert to DataFrame
df_train = pd.DataFrame(train_data, columns=["review", "sentiment"])

# Show sample data
print(df_train.head())
