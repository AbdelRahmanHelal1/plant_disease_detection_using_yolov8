# 🌿 Plant Disease Detection with Telegram Notification
📌 Project Idea
This project detects plant diseases from video input using a YOLO model.
Once a disease is detected, a message is sent to a Telegram bot containing:

The type of disease

Relevant recommendations for treatment or action

# 🤖 Telegram Bot Setup
Create a bot via BotFather on Telegram.

Get the Bot Token and your Chat ID.

Replace the placeholders in your code:

pythonCopyEdit

TOKEN = "YOUR_BOT_TOKEN"

chat_id = "YOUR_CHAT_ID"
# 🎥 Test Video
We used the following video for testing the model:
istockphoto-2150887655-640_adpp_is.mp4

# 🧠 Model
The YOLOv8 trained model file:
best3.pt

The file of training = Download_data.ipynb

# ⚙️ Installation
Install the required packages:

bashCopyEdit

--> pip install -r requirements.txt

# 🚀 Run the Project
Run the main Python file:

bash Copy Edit python 

-- > main.py

# 📬 Output
Once a disease is detected, the bot will automatically send a notification with:

Detected disease name

Recommended treatment steps

# 🎥 Demo
https://www.linkedin.com/posts/abdelrahman-helal-3630a4259_leveragingabryoloabrforabrplantabrdiseaseabrdetectionabrandabrautomatedabrtelegramabralerts-activity-7221863925594640384-_0CR?utm_source=share&utm_medium=member_android
