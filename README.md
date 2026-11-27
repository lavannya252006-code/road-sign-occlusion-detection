# Road Sign Occlusion Detection Using CNN

## Project Overview
This project uses a custom Convolutional Neural Network (CNN) to identify traffic signs and detect whether they are partially obstructed.

## Main Features
- Classification of 43 traffic sign categories.
- Synthetic occlusion generation.
- Dual-output CNN architecture.
- Occlusion detection using binary classification.
- Precision, recall and F1-score evaluation.
- Interactive Streamlit web application.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Dataset
German Traffic Sign Recognition Benchmark (GTSRB).

## Model Architecture
The CNN has two output heads:
1. Traffic sign classification.
2. Binary occlusion detection.

## Application
Users can upload a road sign image and view the model's predictions.

## Note
This is an educational prototype. Its predictions are not validated for real-world road safety.
