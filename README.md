# SignVision AI – ISL Alphabet Recognition

**Real-Time Sign Language Alphabet Recognition Using Computer Vision and Machine Learning**

## 📌 Project Overview

SignVision AI is a real-time sign language alphabet recognition application developed using Python, OpenCV, MediaPipe, and Machine Learning.

The application uses a webcam to capture hand gestures, detects hand landmarks, and predicts the corresponding alphabet from A to Z.

The main objective of this project is to demonstrate how computer vision and machine learning can be used to recognize static hand gestures and support sign language learning.

## 🎯 Objectives

- To develop a real-time hand gesture recognition system.
- To recognize English alphabets (A–Z) using hand gestures.
- To detect hand landmarks using MediaPipe.
- To train a machine learning model using collected gesture samples.
- To display predicted alphabets through a desktop application.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Hand Tracking:** MediaPipe
- **Machine Learning:** Scikit-learn
- **Model:** Random Forest Classifier
- **GUI:** Tkinter
- **Model Storage:** Joblib

## ⚙️ How It Works

1. The webcam captures the user's hand gesture.
2. MediaPipe detects 21 hand landmarks.
3. The landmark coordinates are extracted as numerical features.
4. The trained machine learning model processes these features.
5. The system predicts the corresponding alphabet.
6. The predicted alphabet is displayed on the application screen in real time.

## 📂 Project Structure

```text
SignVisionAI/
│
├── app.py
├── requirements.txt
├── setup_windows.bat
├── run_windows.bat
├── README.md
├── README_HINDI.md
├── PROJECT_OVERVIEW.md
│
├── data/
│   ├── README.txt
│   └── sign_samples.csv
│
├── models/
│   ├── README.txt
│   └── signvision_model.joblib
│
└── LICENSE.txt
```

## 💻 Installation and Setup

### Prerequisites

- Windows 10 or later
- Python 3.11 recommended
- Webcam
- Internet connection for initial dependency installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/rishikamishra0805-bit/SignVision-AI.git
```

### Step 2: Open the Project Folder

```bash
cd SignVision-AI
```

### Step 3: Install Dependencies

Double-click the following file:

```text
setup_windows.bat
```

Alternatively, install dependencies manually:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Application

Double-click:

```text
run_windows.bat
```

The SignVision AI desktop application will open and use the webcam for gesture recognition.

## 🧪 Model Training

The application supports collecting hand gesture samples and training a machine learning model.

The current project contains:

- 26 alphabet classes (A–Z)
- 725 collected samples
- 25 samples per alphabet class
- A trained Random Forest model

The trained model is saved in the `models` directory.

## ✨ Features

- Real-time webcam-based hand detection
- Recognition of English alphabets A–Z
- 21 hand landmark detection
- Machine learning-based classification
- Custom dataset collection
- Model training and saving
- Reloading the saved trained model
- Simple desktop user interface

## ⚠️ Limitations

- The current version recognizes static hand gestures only.
- It is designed for single-hand alphabet recognition.
- Dynamic gestures and complete sign language sentences are not supported.
- Recognition performance may vary depending on lighting, camera quality, and hand position.
- The system is a prototype and is not a complete Indian Sign Language translation system.

## 🔮 Future Scope

- Improve recognition accuracy with a larger dataset.
- Add dynamic gesture recognition.
- Support more sign language vocabulary.
- Develop a web-based interface.
- Explore sentence-level sign language recognition.
- Improve accessibility and user experience.

## 👩‍💻 Developed By

**Rishika Mishra**  
B.Tech – Computer Science and Engineering  
Hindustan College of Science and Technology (HCST)
techiehelp internship project

## 📜 License

This project is intended for educational and academic purposes.
