import cv2
import mediapipe as mp
import csv
import numpy as np
import os

# --- 1. SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Create the CSV file and write the header (optional, but good practice)
csv_file = 'hand_data.csv'
header = ['label'] 
# Generate 63 column names (x1, y1, z1, x2, y2, z2, ..., x21, y21, z21)
for i in range(1, 22):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

# Write header only if the file doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

# --- 2. THE LANDMARK EXTRACTION FUNCTION (Feature Engineering) ---
def extract_landmarks(hand_landmarks, label):
    """Takes MediaPipe landmarks and flattens them into a list of 63 numbers."""
    if not hand_landmarks:
        return None
    
    # Flatten the 21 landmarks (x, y, z) into a single list
    row = [label] # Start with the letter label
    for landmark in hand_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z])
    
    # Simple Normalization/Scaling (Optional but helpful for position independence)
    # The X, Y, Z coordinates are relative to the image size (0 to 1).
    # We can skip complex normalization for the first iteration.
    
    return row

# --- 3. VIDEO CAPTURE LOOP ---
cap = cv2.VideoCapture(0)

print("DATA COLLECTION MODE ACTIVE. Press a key ('a', 'b', etc.) to record data for that sign.")

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Flip the image for a mirrored, more intuitive view
    img = cv2.flip(img, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    current_landmarks = None
    
    if results.multi_hand_landmarks:
        # We assume only one hand is visible for clear data collection
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extract the features for saving
        current_landmarks = extract_landmarks(hand_landmarks, label='TEMP') # 'TEMP' label, will be replaced by key press

    # Display the image with instructions
    cv2.putText(img, "Press a key ('a', 'b') to SAVE 50 FRAMES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Data Collector - Press 'q' to Quit", img)
    
    # --- 4. KEY PRESS LOGIC ---
    key = cv2.waitKey(10)
    
    if key & 0xFF == ord('q'):
        break
        
    # Check if a letter key was pressed and we have landmarks
    if key >= ord('a') and key <= ord('z') and current_landmarks:
        label = chr(key) # Get the character ('a', 'b', 'c', etc.)
        print(f"--- Collecting data for sign: {label.upper()} ---")
        
        # Collect multiple frames (e.g., 50 frames) for better robustness
        frames_to_collect = 50
        
        # Reset the cap for a temporary fast capture
        cap_temp = cv2.VideoCapture(0)
        
        for i in range(frames_to_collect):
            ret, frame = cap_temp.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_temp = hands.process(frame_rgb)
            
            if result_temp.multi_hand_landmarks:
                # Re-extract the landmarks, this time with the correct label
                final_row = extract_landmarks(result_temp.multi_hand_landmarks[0], label)
                
                # Write to CSV
                with open(csv_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(final_row)
            
                # Give visual feedback
                cv2.putText(frame, f"SAVING: {label.upper()} - {i+1}/{frames_to_collect}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
                cv2.imshow("Data Collector - Press 'q' to Quit", frame)
                cv2.waitKey(1) # Refresh display slightly

        cap_temp.release() # Release temporary capture
        print(f"Finished collecting {frames_to_collect} frames for {label.upper()}")
        
cap.release()
cv2.destroyAllWindows()