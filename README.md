## Facial Emotion Recognition using AlexNet

A real-time facial emotion recognition web application built using PyTorch, Flask, and OpenCV. The system uses an AlexNet-based convolutional neural network trained on the FER2013 dataset to classify facial expressions into seven emotion categories.

### Overview

This project implements a facial emotion classification pipeline from scratch, covering model architecture, training, and deployment through a browser-based interface. It supports real-time webcam-based detection with bounding-box face localization and confidence-scored predictions.

### Project Resources

- Project Presentation: see local project file named *AffectNet Paper – Categorical Model-Final*

### Features

- Real-time webcam emotion detection
- Classification across seven facial emotion categories
- AlexNet architecture implemented from scratch in PyTorch
- Browser-based interface built with Flask
- Confidence score displayed alongside predicted emotion
- Face detection with bounding-box overlay

### Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

### Model Architecture

The model is based on AlexNet and consists of:

- 5 convolutional layers
- ReLU activations
- Max pooling layers
- Dropout regularization
- 3 fully connected layers
- Softmax output layer (inference only)

**Input specification**
- Format: RGB
- Resolution: 227 × 227

**Training configuration**
- Loss function: Cross Entropy Loss
- Optimizer: SGD
- Learning rate: 0.01
- Momentum: 0.9
- Weight decay: 0.0005

### Dataset

**FER2013**

Classes:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

### Results

| Metric              | Value  |
|----------------------|-------:|
| Top-1 Accuracy       | 62.76% |
| Top-2 Accuracy       | 80.64% |
| Weighted F1 Score    | 0.62   |
| ROC AUC              | 0.90   |

Two training strategies were compared:

- Standard Cross Entropy (baseline)
- Weighted Cross Entropy

The weighted cross-entropy approach improved recognition of minority classes - particularly Disgust and Fear - at the cost of a slight reduction in overall accuracy.

### Web Interface

The application provides a browser-based interface where users can:

- Start webcam-based detection
- View real-time emotion predictions
- See the predicted emotion label displayed above the detected face
- Stop the webcam session

### Project Structure

```
categorical-interface/
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── app.py
├── webcam.py
├── models.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   ```

2. Move into the project directory
   ```bash
   cd <repository-name>
   ```

3. Create a virtual environment
   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment (Windows)
   ```bash
   .venv\Scripts\activate
   ```

5. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Flask application:
```bash
python app.py
```

Open a browser and navigate to:
```
http://127.0.0.1:5000
```

Click **Detect My Emotion** to begin real-time emotion recognition.

### Model File

The trained model (`best_model.pth`) is not included in this repository, as it exceeds GitHub's file size limit. Place the trained model file in the project root directory before running the application:

```
best_model.pth
```

### Future Improvements

- Incorporate Focal Loss
- Explore ResNet-50 / EfficientNet architectures
- Apply transfer learning
- Add face alignment using MTCNN
- Extend support to AffectNet and RAF-DB datasets
- Improve browser UI
- Add mobile compatibility

### Technologies Used

- Python
- PyTorch
- Flask
- OpenCV
- Pillow
- NumPy

### Author

**Sowmya Sri**
B.Tech, Electronics and Communication Engineering
Indian Institute of Technology Guwahati

If you find this project useful, please consider starring the repository.
