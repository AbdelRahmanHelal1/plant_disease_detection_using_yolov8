import requests
import cv2
import time
from ultralytics import YOLO
import supervision as sv

# Initialize annotators
bound = sv.BoundingBoxAnnotator()
label_ann = sv.LabelAnnotator()

# Telegram Bot Token and Chat ID
TOKEN = "7026186082:AAGOsFhlTY6LyIWLq5duggCf35t8PrvC4QE"
chat_id = "1347052266"

# Load the YOLOv8 model
model = YOLO(r"best3.pt")

# Get class names from the model
class_names = model.names

# Open the video file
cap = cv2.VideoCapture(r"istockphoto-2150887655-640_adpp_is.mp4")

# Dictionary of plant diseases and treatments
plant_diseases_treatments = {
    "Powdery Mildew": {
        "Symptoms": "White or gray powdery spots on leaves, stems, and buds.",
        "Treatment": "Use fungicides containing potassium bicarbonate, neem oil, or sulfur. Ensure good air circulation and avoid overhead watering."
    },
    "Downy Mildew": {
        "Symptoms": "Yellow or white patches on the upper leaf surface, often with fuzzy growth underneath.",
        "Treatment": "Apply fungicides like copper-based products. Improve air circulation and avoid wetting foliage."
    },
    "Early Blight": {
        "Symptoms": "Dark, concentric rings on older leaves, often surrounded by yellow tissue.",
        "Treatment": "Use fungicides containing chlorothalonil or copper. Remove and destroy affected plant parts. Rotate crops and avoid overhead watering."
    },
    "Rust": {
        "Symptoms": "Small, rust-colored pustules on the undersides of leaves.",
        "Treatment": "Apply fungicides with active ingredients like myclobutanil or sulfur. Remove and destroy infected leaves."
    },
    "Leaf Spot": {
        "Symptoms": "Small, water-soaked spots that enlarge and turn brown with a yellow halo.",
        "Treatment": "Use fungicides containing chlorothalonil or mancozeb. Remove and dispose of infected leaves and debris."
    },
    "Leaf Mold": {
        "Symptoms": "Yellow spots on the upper surface of leaves with a grayish, moldy growth on the underside.",
        "Treatment": "Improve air circulation and reduce humidity. Apply fungicides such as copper-based products or chlorothalonil."
    },
    "Root Rot": {
        "Symptoms": "Wilting, yellowing leaves, and rotting roots.",
        "Treatment": "Improve soil drainage, reduce watering, and use fungicides like those containing mefenoxam or phosphorous acid."
    },
    "Leaf Miner": {
        "Symptoms": "Winding, white tunnels inside leaves caused by larvae.",
        "Treatment": "Remove and destroy affected leaves. Use insecticides containing spinosad or neem oil. Introduce beneficial insects such as parasitic wasps."
    },
    "Healthy": {
        "Symptoms": "No symptoms, plants appear vigorous and free of disease.",
        "Treatment": "No treatment"
    }
}




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
