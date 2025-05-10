# Object Detection System - Drishti

Drishti is an object detection system developed to monitor safety compliance in industrial environments. The project uses the YOLOv8 model to detect Personal Protective Equipment (PPE) such as helmets and gloves in real-time video feeds. The system is deployed through a Flask web application that provides live tracking and alerts for workers not wearing the necessary safety gear.


## Project Overview

The main goal of this project is to automatically detect PPE in real-time video streams, improving safety monitoring and compliance. The model was trained on a dataset of over 40,000 images, focusing on detecting helmets and gloves worn by workers in industrial settings.

## Features

- **Real-Time PPE Detection**: Detects helmets and gloves in live video streams with 95% accuracy.
- **Flask Web Application**: Provides a user-friendly interface for live tracking and visualization.
- **Data Preprocessing**: Uses OpenCV to reduce noise and improve model accuracy in challenging environments (low light, high noise).
- **High Precision**: Achieves high precision in detecting safety equipment in diverse industrial settings.
- **Scalable**: Easily deployable to other safety monitoring environments or facilities.

## Technologies Used

- **YOLOv8**: A state-of-the-art object detection model for real-time performance.
- **OpenCV**: For video processing and image enhancements (e.g., noise reduction).
- **Flask**: For building the web application for live video streaming and detection.
- **Python**: Primary programming language used for training and deploying the model.
- **TensorFlow/PyTorch**: For model training and optimization.
- **NumPy & Pandas**: For data manipulation and analysis.

## Installation Instructions

### Prerequisites

Before running the project, ensure that you have the following installed:

- Python 3.x
- pip (Python package manager)
- Git

### Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/rv-raghav/ObjectDetection.git
cd ObjectDetection
```
- pip install -r requirements.txt
