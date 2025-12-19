import cv2
import mediapipe as mp
import csv
import numpy as np
import os
from tqdm import tqdm # Library for progress bars (install with: pip install tqdm)

# --- 0. INSTALL tqdm FOR A PROGRESS BAR ---
# Run this once in your terminal: pip install tqdm

# --- 1. SETUP ---
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

csv_file = 'hand_data.csv'
BASE_DIR = 'C:\\Users\\nadir\\OneDrive\\Desktop\\SignLanguageProject\\archive' # <-- CRITICAL: Change this if your folder name is different!

# --- 2. Landmark Extraction Function (from data_collector.py) ---
def extract_landmarks(hand_landmarks, label):
    """Flattens the 21 landmarks (x, y, z) into a list of 63 numbers."""
    if not hand_landmarks:
        return None
    
    row = [label] # Start with the letter label
    for landmark in hand_landmarks.landmark:
        # Scale coordinates (optional, but good practice for consistency)
        row.extend([landmark.x, landmark.y, landmark.z])
    return row

# --- 3. Processing Logic ---
print("Starting integration of external dataset...")
total_processed = 0

# Loop through every sub-folder (A, B, C, etc.)
# We use os.walk to explore the directories
for root, dirs, files in os.walk(BASE_DIR):
    # The label is the name of the folder (e.g., 'A', 'B', 'C')
    label = os.path.basename(root).lower()
    
    # Check if the folder name is a single letter (our label)
    if len(label) == 1 and label.isalpha():
        print(f"\nProcessing sign: {label.upper()}")
        
        # Use tqdm to show a professional progress bar
        for file_name in tqdm(files):
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(root, file_name)
                
                # Read the image
                image = cv2.imread(file_path)
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image to find hands
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    # We assume only one hand is in the image
                    landmarks = results.multi_hand_landmarks[0]
                    final_row = extract_landmarks(landmarks, label)
                    
                    # Write to CSV
                    with open(csv_file, mode='a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(final_row)
                    total_processed += 1

print(f"\nIntegration Complete! Total new landmarks added: {total_processed}")
print("Your training file is now ready for re-training.")