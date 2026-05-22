# Image Classification App 🖼️

Classifies any image into 1000 categories
using ResNet50 pretrained on ImageNet.

## Live Demo
[Click here](https://image-classifier-app-75zbcwvw8e5bfkubt63gui.streamlit.app)

## Features
- Upload any image for instant classification
- Top-K predictions with confidence scores
- Full probability distribution chart
- Image preprocessing visualization
- ResNet50 architecture breakdown
- Model comparison table

## Model Details
- Architecture: ResNet50
- Parameters: 25.6M
- Training: ImageNet (1.2M images)
- Top-5 Accuracy: 93.7%
- Categories: 1000

## Key Concepts
- Transfer Learning
- Residual Connections
- Skip Connections
- ImageNet benchmark

## Tools Used
- Python, PyTorch, torchvision
- Streamlit, Pillow, Matplotlib

## How to Run Locally
pip install streamlit torch torchvision Pillow numpy matplotlib pandas
streamlit run app.py
