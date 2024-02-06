from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
from model import model
import torch

app = Flask(__name__)

def predict_segment(image_data):
    # Decode base64-encoded image data
    decoded_image = base64.b64decode(image_data)

    # Convert to NumPy array
    nparr = np.frombuffer(decoded_image, np.uint8)

    # Read the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cur_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cur_image = cv2.resize(cur_image, (1088,1920))

    # Convert the image to black and white
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = torch.tensor(cur_image, dtype=torch.float)
    image = image.permute(2, 0, 1)
    image=image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    segmented_image = output.squeeze()
    segmented_image=segmented_image.detach().cpu().numpy()
    segmented_image = 1 - segmented_image

    # Convert the processed image back to base64 encoding
    _, encoded_image = cv2.imencode('.png', segmented_image.astype(np.uint8))
    return base64.b64encode(encoded_image).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        original_image = base64.b64encode(file.read()).decode('utf-8')
        processed_image = predict_segment(original_image)
        return render_template('index.html', original_image=original_image, processed_image=processed_image)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
