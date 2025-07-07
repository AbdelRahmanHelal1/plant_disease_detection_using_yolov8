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
