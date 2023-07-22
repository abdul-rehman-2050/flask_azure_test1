from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import base64
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from skimage import feature





app = Flask(__name__)

# Helper function to decode base64 image data
def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))


@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    try:
        # Get the JSON data from the request
        data = request.json
        base64_image = data['inputs'][0]['data']['image']['base64']

        # Decoding base64 and converting to grayscale
        image = decode_base64_image(base64_image).convert('L')

        # Resize the image to 96x96
        image = image.resize((96, 96), Image.LANCZOS)

        # Load and preprocess 'img1.jpg'
        img1 = Image.open('img1.jpg').convert('L')
        img1 = img1.resize((96, 96), Image.LANCZOS)
        img1 = np.array(img1)

        # Convert the images to grayscale
        image_gray = np.array(image)
        img1_gray = np.array(img1)

        # Convert both grayscale images to binary edge images using Canny edge detection
        image_edges = cv2.Canny(image_gray, 50, 100)
        img1_edges = cv2.Canny(img1_gray, 50, 100)

        # Convert images to NumPy arrays for similarity calculation
        image_np = image_edges.flatten().reshape(1, -1)
        img1_np = img1_edges.flatten().reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(image_np, img1_np)[0][0]

        # Return the similarity as JSON response
        return jsonify({'similarity': similarity})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Check if a file named 'image' is sent in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image sent'}), 400
        
        # Read the image from the request
        image_file = request.files['image']
        image_np = np.fromfile(image_file, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Resize the image to 100x100
        resized_image = cv2.resize(image, (100, 100))
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Convert the grayscale image to binary (black and white)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Encode the binary image as a base64 string to send it back in the response
        _, encoded_image = cv2.imencode('.png', binary_image)
        encoded_image_base64 = encoded_image.tobytes().decode('utf-8')
        
        return jsonify({'binary_image': encoded_image_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
