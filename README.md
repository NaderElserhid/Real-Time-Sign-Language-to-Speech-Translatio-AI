# Real-Time Sign Language to Speech Translator 🤟🔊

This project is an AI-powered system designed to bridge the communication gap for the hearing-impaired. It uses **Computer Vision** and **Machine Learning** to recognize American Sign Language (ASL) gestures from a webcam and convert them into real-time speech.

## 🚀 Key Features
- **Landmark Detection:** Uses MediaPipe to track 21 hand joints in 3D space.
- **High Accuracy:** Achieved **98.44% test accuracy** using a Random Forest Classifier.
- **Text-to-Speech:** Integrated verbal output using `pyttsx3`.
- **Hybrid Dataset:** Trained on a mix of custom-collected data and the Kaggle ASL Alphabet dataset (~64,000 samples).

## 🛠️ Technical Architecture
Instead of processing raw pixels (which is slow), this project uses **Feature Engineering**:
1. **Input:** OpenCV captures the webcam feed.
2. **Processing:** MediaPipe extracts 63 coordinates (x, y, z for 21 landmarks).
3. **AI Model:** A Random Forest model predicts the letter based on these coordinates.
4. **Output:** The result is displayed on screen and spoken via the system's speakers.



## 📁 Project Structure
- `data_collector.py`: Tool for capturing custom hand gestures.
- `dataset_integrator.py`: Processes the Kaggle image dataset into coordinates.
- `train_model.py`: Trains the AI and saves the model.
- `sign_to_speech.py`: The final real-time application.

## ⚙️ Setup & Installation
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/REPO_NAME.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python sign_to_speech.py`

*Note: The .pkl model and .csv data files are excluded due to size limits. Run `train_model.py` to generate a local model.*
