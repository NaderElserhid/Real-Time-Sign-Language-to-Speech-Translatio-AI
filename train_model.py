import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# --- 1. Load Data ---
print("1. Loading Data...")
# Load the CSV file created in Phase 4
df = pd.read_csv('hand_data.csv')

# --- 2. Separate Features (X) and Labels (y) ---
# The label is the first column ('label'), and the rest are the 63 features
X = df.drop('label', axis=1).values # Features (the 63 coordinates)
y = df['label'].values             # Labels (the letter, e.g., 'a', 'b')

# --- 3. Splitting the Data ---
# 80% for training the model, 20% for testing how well it generalizes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Total samples collected: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- 4. Training the Model (Random Forest) ---
print("\n4. Training Random Forest Classifier...")
# We use a Random Forest because it handles high-dimensional coordinate data very well.
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)

# --- 5. Evaluation ---
print("5. Evaluating Model Performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Model Accuracy on Test Data: {accuracy * 100:.2f}% ---")

# --- 6. Saving the Model ---
# We save the trained model object so we don't have to train it again every time.
model_filename = 'sign_language_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
    
print(f"Model successfully trained and saved as {model_filename}")