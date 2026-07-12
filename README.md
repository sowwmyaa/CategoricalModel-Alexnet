# 🎭 Facial Emotion Recognition using AlexNet

A real-time **Facial Emotion Recognition** web application built using **PyTorch**, **Flask**, and **OpenCV**. The system uses an **AlexNet-based Convolutional Neural Network** trained on the **FER2013** dataset to classify facial expressions into seven emotion categories.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch">
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?logo=flask">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv">
</p>

---

## 📄 Project Resources

- 📑 **Project Presentation:** [View Presentation](file:///C:/Users/Sowmya/Desktop/AffectNet%20Paper-Categorical%20model.pdf)

## 📌 Features

- 🎥 Real-time webcam emotion detection
- 😊 Detects **7 facial emotions**
- 🧠 AlexNet implemented from scratch in PyTorch
- 🌐 Browser-based interface using Flask
- 📦 Simple and lightweight interface
- 📊 Displays predicted emotion with confidence score
- 🟦 Face detection with bounding box

---

## 😊 Supported Emotions

- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😀
- Neutral 😐
- Sad 😢
- Surprise 😲

---

## 🏗️ Model Architecture

The model is based on **AlexNet** and consists of:

- 5 Convolutional Layers
- ReLU Activations
- Max Pooling Layers
- Dropout Regularization
- 3 Fully Connected Layers
- Softmax Prediction (during inference)

Input Image:

- RGB
- 227 × 227

Loss Function:

- Cross Entropy Loss

Optimizer:

- SGD
- Learning Rate = 0.01
- Momentum = 0.9
- Weight Decay = 0.0005

---

## 📂 Dataset

Dataset used:

**FER2013**

Classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## 📈 Results

| Metric | Value |
|--------|------:|
| Top-1 Accuracy | **62.76%** |
| Top-2 Accuracy | **80.64%** |
| Weighted F1 Score | **0.62** |
| ROC AUC | **0.90** |

The project also compares:

- Standard Cross Entropy (Baseline)
- Weighted Cross Entropy

Weighted Cross Entropy improved minority-class recognition (especially **Disgust** and **Fear**) but reduced the overall accuracy.

---

## 💻 Web Interface

The application provides a simple browser interface where users can:

- Start webcam detection
- Detect emotions in real time
- View the predicted emotion above the detected face
- Stop the webcam

---

## 📁 Project Structure

```text
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

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Move into the project

```bash
cd <repository-name>
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

Click **Detect My Emotion** to start real-time emotion recognition.

---

## ⚠️ Model File

The trained model (`best_model.pth`) is **not included** in this repository because it exceeds GitHub's file size limit.

Place the trained model in the project root before running the application.

```
best_model.pth
```

---

## 🔮 Future Improvements

- Focal Loss
- ResNet-50 / EfficientNet
- Transfer Learning
- Face Alignment using MTCNN
- Support for AffectNet and RAF-DB
- Improved browser UI
- Mobile compatibility

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Flask
- OpenCV
- Pillow
- NumPy

---

## 👩‍💻 Author

**Sowmya Sri**

B.Tech, Electronics and Communication Engineering  
Indian Institute of Technology Guwahati

---

## ⭐ If you found this project useful

Please consider giving the repository a **Star ⭐**.
