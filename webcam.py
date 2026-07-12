import cv2
import torch
from PIL import Image
from torchvision import transforms
from models import AlexNet

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Emotion Labels
# -----------------------------
classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -----------------------------
# Load Model
# -----------------------------
model = AlexNet(num_classes=7).to(device)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()

# -----------------------------
# Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Webcam
# -----------------------------
camera = cv2.VideoCapture(0)


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = Image.fromarray(face)

            face = transform(face)

            face = face.unsqueeze(0).to(device)

            with torch.no_grad():

                outputs = model(face)

                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted = torch.max(probabilities, 1)

            emotion = classes[predicted.item()]
            confidence = confidence.item() * 100

            # Face box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

            # Label
            label = f"{emotion} ({confidence:.1f}%)"

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            label_y = max(y - 10, th + 10)

            cv2.rectangle(
                frame,
                (x, label_y - th - 10),
                (x + tw + 10, label_y + 5),
                (40, 40, 40),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )
