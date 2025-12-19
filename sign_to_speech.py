import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3

# --- 1. Setup ---
# Load the trained model
try:
    with open('sign_language_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file 'sign_language_model.pkl' not found.")
    print("Please run train_model.py successfully before running this script.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
# Optional: Slow down the speech rate slightly for better clarity
tts_engine.setProperty('rate', 150) 

# --- 2. Live Prediction Variables ---
last_spoken_time = time.time()
prediction_history = []
speech_delay = 2.0  # Time in seconds between speaking new words
min_history_length = 10 # Number of recent predictions to consider
prediction_threshold = 0.8 # 80% confidence needed to speak

# --- 3. Landmark Extraction Function ---
def extract_landmarks(hand_landmarks):
    """Flattens the 21 landmarks (x, y, z) into a single 63-element array."""
    if not hand_landmarks:
        return None
    
    row = []
    for landmark in hand_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z])
    
    # Return as a numpy array with shape (1, 63) suitable for model prediction
    return np.array(row).reshape(1, -1)

# --- 4. Video Capture Loop ---
cap = cv2.VideoCapture(0)

print("Sign-to-Speech Application Running. Press 'q' to Quit.")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Flip for a natural mirror view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predicted_letter = ""
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extract features and predict
        features = extract_landmarks(hand_landmarks)
        
        if features is not None:
            prediction = model.predict(features)[0] # Predict the letter
            
            # --- 5. Smoothing and Stabilization Logic ---
            # Add the current prediction to the history
            prediction_history.append(prediction)
            
            # Keep history short to only look at recent frames
            if len(prediction_history) > min_history_length:
                prediction_history.pop(0)
            
            # Find the most frequent prediction in the history
            from collections import Counter
            most_common = Counter(prediction_history).most_common(1)
            
            if most_common:
                stable_prediction, count = most_common[0]
                
                # Check if the most common prediction meets the confidence threshold
                confidence = count / len(prediction_history)
                
                if confidence >= prediction_threshold:
                    predicted_letter = stable_prediction.upper()
                    
                    # --- 6. Text-to-Speech Output ---
                    current_time = time.time()
                    
                    # Only speak if enough time has passed since the last spoken word
                    # AND the current letter is different from the last spoken letter
                    if predicted_letter and (current_time - last_spoken_time) > speech_delay:
                        
                        # Prevent the engine from freezing the video feed
                        tts_engine.say(predicted_letter)
                        tts_engine.runAndWait() 
                        
                        last_spoken_time = current_time
                        
                        # Clear history to avoid re-speaking immediately
                        prediction_history.clear() 

    # Display the result on the screen
    cv2.putText(frame, f"Sign: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Sign Language to Speech (Final Project)", frame)
    
    # Exit loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()