import cv2
import numpy as np
import torch
from collections import deque, Counter

from model import Sign3DCNN


torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("model_3dcnn.pth", map_location=DEVICE)

model = Sign3DCNN(num_classes=checkpoint["num_classes"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

CLASSES = checkpoint["classes"]
SEQUENCE_LENGTH = checkpoint["sequence_length"]
IMG_SIZE = checkpoint["img_size"]

cap = cv2.VideoCapture(0)

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=8)

current_text = ""
detected_letter = ""
current_conf = 0.0
frame_count = 0

PREDICT_EVERY_N_FRAMES = 2
CONF_THRESHOLD = 0.70


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.flip(frame, 1)

    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    frame_buffer.append(rgb)

    if len(frame_buffer) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY_N_FRAMES == 0:
        clip = np.array(frame_buffer, dtype=np.float32)
        clip = np.transpose(clip, (3, 0, 1, 2))
        clip = np.expand_dims(clip, axis=0)
        clip = torch.tensor(clip, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            output = model(clip)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        current_conf = conf.item()

        if current_conf >= CONF_THRESHOLD:
            prediction_buffer.append(pred.item())
            final_pred = Counter(prediction_buffer).most_common(1)[0][0]
            detected_letter = CLASSES[final_pred]
        else:
            detected_letter = ""

    cv2.putText(
        frame, f"Detected Letter: {detected_letter}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2
    )

    cv2.putText(
        frame, f"Confidence: {current_conf:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

    cv2.putText(
        frame, f"Text: {current_text}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 255), 2
    )

    cv2.putText(
        frame, "ENTER - add letter | BACKSPACE - delete | C - clear | Q - quit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    cv2.imshow("Letter Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Enter
        if detected_letter != "":
            current_text += detected_letter

    elif key == 8:  # Backspace
        current_text = current_text[:-1]

    elif key == ord("c"):
        current_text = ""
        prediction_buffer.clear()

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()