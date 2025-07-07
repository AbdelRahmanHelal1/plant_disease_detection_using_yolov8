import requests
import cv2
import time
from ultralytics import YOLO
import supervision as sv

from plant_diseases_treatments import plant_diseases_treatments

# Initialize annotators
bound = sv.BoundingBoxAnnotator()
label_ann = sv.LabelAnnotator()

# Telegram Bot Token and Chat ID
TOKEN = "TOKEN FOR YOUR BOT"
chat_id = "CHAT ID FOR YOUR BOT"

# Load the YOLOv8 model
model = YOLO(r"best3.pt")

# Get class names from the model
class_names = model.names

# Open the video file
cap = cv2.VideoCapture(r"istockphoto-2150887655-640_adpp_is.mp4")


def send_image(bot_token, chat_id, image_path):
    """
    Sends an image to a Telegram chat.

    Parameters:
    - bot_token: The token for the Telegram bot.
    - chat_id: The ID of the Telegram chat to send the image to.
    - image_path: The file path of the image to send.
    """
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    files = {'photo': open(image_path, 'rb')}
    data = {'chat_id': chat_id}
    response = requests.post(url, files=files, data=data)
    return response.json()


def send_image_and_treatment(annotated_image_path, detected_diseases):
    """
    Send image and treatment information to Telegram.

    Parameters:
    - annotated_image_path: The path to the annotated image.
    - detected_diseases: A list of detected diseases and their treatments.
    """
    send_image(TOKEN, chat_id, annotated_image_path)
    for disease, treatment in detected_diseases.items():
        message = f"Disease: {disease}\nTreatment: {treatment}"
        url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
        data = {'chat_id': chat_id, 'text': message}
        requests.post(url, data=data)


# Timer setup
start_time = time.time()
send_interval = 10  # 10 minutes in seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 600))

    # Perform prediction
    result = model.predict(frame)
    detection = sv.Detections.from_ultralytics(result[0])

    # Annotate frame with bounding boxes and labels
    ann_frame = bound.annotate(
        scene=frame.copy(), detections=detection
    )

    detected_diseases = {}
    for i, det in enumerate(detection.xyxy):
        x, y, w, z = det
        class_id = int(detection.class_id[i])
        label = class_names[class_id]
        treatment = plant_diseases_treatments[label] .get(label, {}).get("Treatment", plant_diseases_treatments[label]["Treatment"])
        detected_diseases[label] = treatment
        print(f"Disease: {label}, Treatment: {treatment}")
        ann_frame = cv2.putText(ann_frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save annotated frame as image
    annotated_image_path = "annotated_image.jpg"
    cv2.imwrite(annotated_image_path, ann_frame)

    # Send image and treatment info every 10 minutes
    current_time = time.time()
    if current_time - start_time >= send_interval:
        send_image_and_treatment(annotated_image_path, detected_diseases)
        start_time = current_time  # Reset the timer

    # Display image using cv2.imshow (only if supported)
    cv2.imshow("image", ann_frame)
    if cv2.waitKey(1) == ord("s"):
        break

cap.release()
cv2.destroyAllWindows()
