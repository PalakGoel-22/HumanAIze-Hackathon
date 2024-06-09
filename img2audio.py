import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, send_file
import google.generativeai as genai
from werkzeug.utils import secure_filename
from PIL import Image
from gtts import gTTS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure the Google Generative AI using the environment variable
GOOGLE_API_KEY = os.getenv('API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("API_KEY environment variable is not set")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

# Define a list to store uploaded files temporarily
uploaded_files = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Append the file to the uploaded_files list
        uploaded_files.append(file)

        # Open the image file using PIL
        img = Image.open(file.stream)

        # Use the AI model to generate content from the image
        response = model.generate_content(["Write a 1 sentence whatever is there in the picture", img], stream=True)
        response.resolve()
        
        detailed_response = model.generate_content(img)
        
        # Generate audio from the text responses
        text_response = response.text
        text_response2 = detailed_response.text

        tts = gTTS(text_response)
        tts2 = gTTS(text_response2)

        audio_file_path = os.path.join('static', 'audio', 'response.mp3')
        audio_file_path2 = os.path.join('static', 'audio', 'response2.mp3')

        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
        tts.save(audio_file_path)

        os.makedirs(os.path.dirname(audio_file_path2), exist_ok=True)
        tts2.save(audio_file_path2)
        
        # Return both generated text, detailed response, and audio file URLs
        return jsonify({
            'text': text_response, 
            'detailed_response': text_response2,
            'audio_url1': f'/static/audio/response.mp3',
            'audio_url2': f'/static/audio/response2.mp3'
        })
    return jsonify({'error': 'Unknown error'})

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    return send_file(os.path.join('static', 'audio', filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # Use a safe port
