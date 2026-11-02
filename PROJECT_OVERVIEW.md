# SignVision AI — Project Overview

## Title
SignVision AI: AI-Powered Indian Sign Language Hand-Pose Recognition Prototype

## Objective
To develop a desktop prototype that captures hand landmarks from a webcam and classifies user-trained static sign labels into text.

## Technology Stack
- Python 3.11
- OpenCV — webcam frame capture
- MediaPipe Hands — 21 hand landmarks
- NumPy — landmark feature normalization
- Scikit-learn Random Forest — classifier training and prediction
- Tkinter — desktop GUI
- Joblib — saving/loading trained model

## Workflow
Webcam frame → MediaPipe hand detection → 21 landmarks → wrist-relative normalization → 63 numeric features → Random Forest classifier → label + confidence → sentence builder.

## Current Scope
A–Z and common word labels are provided as trainable classes. The user must collect examples for the labels and train the model locally. The app currently handles one hand and static poses.

## Future Scope
- Pretrained ISL dataset and model
- Dynamic sign recognition using temporal sequence models
- Two-hand gesture recognition
- Facial expression and body-pose features
- ISL grammar-aware sentence translation
- Mobile deployment
